# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The fresh-clone contract, tested so it cannot rot:
#
# 1. `pip install -e '.[all]'` pulls in every runtime dependency of every
#    tool -- including kokoro-onnx for TTS, which once lived outside every
#    extra and left a fresh clone's `speak` broken with an install hint.
# 2. The .onnx/.bin weights are never in the repo (gitignored), so the
#    first `speak` on a fresh machine must download them itself.
# 3. _download() must tolerate a partial download: resume a .part file
#    when the server honours Range, restart when it does not, and never
#    promote a truncated body.
#
# (1) reads installed package metadata -- it does not install anything.
# (2) and (3) fake the network and the engine, so no request leaves the
# process and the real 325 MB model is never loaded by a test: the engine
# import is intercepted through sys.modules, because _get_engine() does
# `from kokoro_onnx import Kokoro` locally and a module-attribute patch
# would not touch it (that mistake once loaded the real model mid-suite).

import os
import sys
import types
import urllib.error
import urllib.request
from importlib.metadata import metadata
from unittest import mock

import pytest

from onit.mcp.servers.tasks.tools import tts


class TestFreshCloneInstallsTtsRuntime:
    """Every runtime import of the speak tool must be pip-installable
    from the extras a fresh clone is told to install."""

    def test_kokoro_onnx_is_in_the_tts_extra(self):
        requires = metadata("onit").get_all("Requires-Dist") or []
        tts_reqs = [r for r in requires if "extra == 'tts'" in r or 'extra == "tts"' in r]
        assert tts_reqs, "no Requires-Dist carries the tts extra -- is it wired in pyproject.toml?"
        names = set()
        for r in tts_reqs:
            # Parse, don't string-split: a PEP 508 specifier like
            # "kokoro-onnx>=0.6.1" needs a real requirement parser.
            from packaging.requirements import Requirement

            names.add(Requirement(r).name.lower())
        assert "kokoro-onnx" in names, names

    def test_tts_extra_is_reachable_from_all(self):
        requires = metadata("onit").get_all("Requires-Dist") or []
        all_self = [r for r in requires if r.strip().lower().startswith("onit[")]
        joined = "; ".join(all_self)
        assert "tts" in joined, f"onit[all] does not pull the tts extra: {all_self}"


class TestWeightsAutoDownload:
    def test_weights_dir_is_outside_the_repo(self):
        # The whole point of the gitignore: weights live in $HOME, never
        # in the tree, so a clone never carries or needs them.
        repo = os.path.abspath(os.path.join(os.path.dirname(tts.__file__), "..", "..", "..", ".."))
        assert not os.path.abspath(tts.KOKORO_MODEL_DIR).startswith(repo), (
            f"weights dir {tts.KOKORO_MODEL_DIR} resolved inside the repo"
        )

    def test_ensure_weights_downloads_then_reuses(self, tmp_path):
        calls = []

        def fake_download(url, dest):
            calls.append(url)
            with open(dest, "wb") as f:
                f.write(b"placeholder-bytes")

        with (
            mock.patch.object(tts, "KOKORO_MODEL_DIR", str(tmp_path)),
            mock.patch.object(tts, "_download", side_effect=fake_download),
        ):
            mp, vp = tts.ensure_weights()
            assert calls == [tts.MODEL_URL, tts.VOICES_URL]
            assert os.path.isfile(mp) and os.path.getsize(mp) > 0
            assert os.path.isfile(vp) and os.path.getsize(vp) > 0
            # Second call: cached, no more downloads.
            mp2, vp2 = tts.ensure_weights()
            assert (mp2, vp2) == (mp, vp)
            assert len(calls) == 2

    def test_corrupt_weight_file_is_redownloaded(self, tmp_path):
        """An empty cached file must not be served as a model."""
        calls = []

        def fake_download(url, dest):
            calls.append(url)
            with open(dest, "wb") as f:
                f.write(b"placeholder-bytes")

        model_path = os.path.join(str(tmp_path), tts.MODEL_FILE)
        with (
            mock.patch.object(tts, "KOKORO_MODEL_DIR", str(tmp_path)),
            mock.patch.object(tts, "_download", side_effect=fake_download),
        ):
            # Seed a zero-length model file: the cache check must reject it.
            open(model_path, "wb").close()
            mp, vp = tts.ensure_weights()
            assert os.path.getsize(mp) > 0
            assert calls.count(tts.MODEL_URL) == 1

    def test_load_failure_triggers_one_redownload(self, tmp_path):
        """A corrupt cache that passes the size check must be dropped and
        fetched once more, then surface as an error -- not loop forever.
        The engine import is faked at sys.modules level because the real
        import happens inside _get_engine()."""

        class Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("bad weights")

        fake_pkg = types.ModuleType("kokoro_onnx")
        fake_pkg.Kokoro = Boom
        calls = []

        def counting_download(url, dest):
            calls.append(url)
            with open(dest, "wb") as f:
                f.write(b"placeholder-bytes")

        with (
            mock.patch.object(tts, "KOKORO_MODEL_DIR", str(tmp_path)),
            mock.patch.object(tts, "_download", side_effect=counting_download),
            mock.patch.dict(sys.modules, {"kokoro_onnx": fake_pkg}),
            mock.patch.object(tts, "_ENGINE", None),
        ):
            with pytest.raises(tts.TTSError, match="Failed to load the Kokoro model"):
                tts._get_engine()
            # Both files fetched on the first attempt and both again on
            # the retry: two downloads per file, one retry, no loop.
            assert calls.count(tts.MODEL_URL) == 2, calls
            assert calls.count(tts.VOICES_URL) == 2, calls


def _respond(status, headers, chunks):
    """A urlopen stand-in whose response hands out *chunks* once each,
    then b\"\" -- read() must go empty or _download's copy loop never ends
    (an always-truthy read hung the suite once already).

    *headers* is looked up case-insensitively, like HTTPMessage.get: the
    production code reads Content-Length via .get on the real response
    headers, and a plain dict of lowercase keys would make a correct
    implementation look wrong (an earlier draft of this test failed that
    way and briefly framed a truncation bug that did not exist)."""

    class Resp:
        pass

    Resp.status = status
    body = iter(chunks)

    def read(self, n=-1):
        try:
            return next(body)
        except StopIteration:
            return b""

    Resp.read = read

    class _Headers(dict):
        def get(self, key, default=None):
            return super().get(key.lower(), default)

    Resp.headers = _Headers({k.lower(): v for k, v in headers.items()})

    class Ctx:
        def __enter__(self):
            return Resp()

        def __exit__(self, *a):
            return False

    return Ctx()


class TestDownloadResilience:
    def test_resume_appends_on_206(self, tmp_path):
        dest = os.path.join(str(tmp_path), "f.bin")
        with open(dest + ".part", "wb") as f:
            f.write(b"HEAD")

        def fake_urlopen(req, timeout=None):
            assert req.headers.get("Range") == "bytes=4-"
            # RFC 7233: a 206's Content-Length counts the bytes of THIS
            # response body -- the remaining 4 -- not the whole file. The
            # code checks offset against CL + resume, which is right.
            return _respond(206, {"content-length": "4"}, [b"TAIL"])

        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            tts._download("http://x/f.bin", dest)
        with open(dest, "rb") as f:
            assert f.read() == b"HEADTAIL"

    def test_no_range_support_restarts(self, tmp_path):
        dest = os.path.join(str(tmp_path), "f.bin")
        with open(dest + ".part", "wb") as f:
            f.write(b"STALE")

        # A plain 200 means the server ignored the Range header: the file
        # must be rewritten from scratch, not appended to.
        with mock.patch.object(
            urllib.request,
            "urlopen",
            lambda req, timeout=None: _respond(200, {"content-length": "5"}, [b"fresh"]),
        ):
            tts._download("http://x/f.bin", dest)
        with open(dest, "rb") as f:
            assert f.read() == b"fresh"

    def test_truncated_body_is_rejected(self, tmp_path):
        dest = os.path.join(str(tmp_path), "f.bin")

        with mock.patch.object(
            urllib.request,
            "urlopen",
            lambda req, timeout=None: _respond(200, {"content-length": "100"}, [b"short"]),
        ):
            with pytest.raises(tts.TTSError, match="truncated"):
                tts._download("http://x/f.bin", dest)
        assert not os.path.exists(dest + ".part"), "a truncated .part must not be promoted"

    def test_http_failure_raises_tts_error(self, tmp_path):
        def boom(req, timeout=None):
            raise urllib.error.URLError("no network")

        with mock.patch.object(urllib.request, "urlopen", boom):
            with pytest.raises(tts.TTSError, match="no network"):
                tts._download("http://x/f.bin", os.path.join(str(tmp_path), "f.bin"))


class TestMissingDependencyMessage:
    def test_missing_kokoro_onnx_error_names_the_extra(self):
        """The error a fresh-clone user sees must point at the extras fix,
        not at a bare package name they then have to reason about."""
        import builtins

        real_import = builtins.__import__

        def no_kokoro(name, *a, **k):
            if name == "kokoro_onnx":
                raise ImportError("No module named 'kokoro_onnx'")
            return real_import(name, *a, **k)

        with (
            mock.patch.object(tts, "_ENGINE", None),
            mock.patch("builtins.__import__", side_effect=no_kokoro),
        ):
            with pytest.raises(tts.TTSError, match=r"\.\[tts\]"):
                tts._get_engine()
