"""Tests for container_launcher — no live Docker required."""

import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, __file__.rsplit("/test/", 1)[0])

from container_launcher import (  # noqa: E402
    DATA_VOLUME,
    IMAGE_TAG,
    build_run_command,
    strip_container_flag,
    strip_launcher_args,
)


def test_strip_container_flag_removes_flag():
    argv = ["--web", "--container", "--web-port", "9000"]
    assert strip_container_flag(argv) == ["--web", "--web-port", "9000"]


def test_strip_container_flag_no_op_when_absent():
    argv = ["--web", "--web-port", "9000"]
    assert strip_container_flag(argv) == argv


def test_strip_launcher_args_removes_gpus_and_mounts():
    argv = [
        "--container",
        "--container-gpus", "all",
        "--container-mount", "/host/docs:/home/onit/documents:ro",
        "--container-mount", "/host/data:/data:ro",
        "--web",
    ]
    assert strip_launcher_args(argv) == ["--web"]


def test_strip_launcher_args_handles_equals_form():
    argv = ["--container", "--container-gpus=all", "--web"]
    assert strip_launcher_args(argv) == ["--web"]


def test_build_run_command_terminal_mode_uses_tty():
    cmd = build_run_command(
        "docker", [], config_mounts=[], secret_env=[]
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "-it" in cmd
    assert IMAGE_TAG in cmd


def test_build_run_command_server_mode_no_tty():
    cmd = build_run_command(
        "docker", ["--web"], config_mounts=[], secret_env=[]
    )
    assert "-it" not in cmd
    assert "-i" in cmd


def test_build_run_command_applies_hardening():
    cmd = build_run_command(
        "docker", [], config_mounts=[], secret_env=[]
    )
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--pids-limit" in cmd
    assert "--memory" in cmd


def test_build_run_command_web_maps_9000():
    cmd = build_run_command(
        "docker", ["--web"], config_mounts=[], secret_env=[]
    )
    # The port-map pair appears as adjacent args.
    i = cmd.index("-p")
    assert cmd[i + 1] == "9000:9000"


def test_build_run_command_a2a_maps_9001():
    cmd = build_run_command(
        "docker", ["--a2a"], config_mounts=[], secret_env=[]
    )
    assert "9001:9001" in cmd


def test_build_run_command_honors_custom_web_port():
    cmd = build_run_command(
        "docker", ["--web", "--web-port", "9500"], config_mounts=[], secret_env=[]
    )
    assert "9500:9500" in cmd
    assert "9000:9000" not in cmd


def test_build_run_command_honors_custom_a2a_port_equals_form():
    cmd = build_run_command(
        "docker", ["--a2a", "--a2a-port=9100"], config_mounts=[], secret_env=[]
    )
    assert "9100:9100" in cmd


def test_build_run_command_passes_gpus():
    cmd = build_run_command(
        "docker", [], config_mounts=[], secret_env=[], gpus="all"
    )
    i = cmd.index("--gpus")
    assert cmd[i + 1] == "all"


def test_build_run_command_no_gpus_by_default():
    cmd = build_run_command(
        "docker", [], config_mounts=[], secret_env=[]
    )
    assert "--gpus" not in cmd


def test_build_run_command_passes_extra_mounts():
    cmd = build_run_command(
        "docker",
        [],
        config_mounts=[],
        secret_env=[],
        mounts=["/host/docs:/home/onit/documents:ro", "/host/data:/data:ro"],
    )
    assert "/host/docs:/home/onit/documents:ro" in cmd
    assert "/host/data:/data:ro" in cmd
    # Each extra mount must be preceded by -v.
    i1 = cmd.index("/host/docs:/home/onit/documents:ro")
    i2 = cmd.index("/host/data:/data:ro")
    assert cmd[i1 - 1] == "-v"
    assert cmd[i2 - 1] == "-v"


def test_build_run_command_no_ports_in_terminal_mode():
    cmd = build_run_command(
        "docker", [], config_mounts=[], secret_env=[]
    )
    assert "-p" not in cmd


def test_build_run_command_mounts_data_volume():
    cmd = build_run_command(
        "docker", [], config_mounts=[], secret_env=[]
    )
    # Mount arg is `{volume}:/home/onit/data`.
    assert any(
        a.startswith(f"{DATA_VOLUME}:") and a.endswith(":/home/onit/data") is False
        and "/home/onit/data" in a
        for a in cmd
    ) or f"{DATA_VOLUME}:/home/onit/data" in cmd


def test_build_run_command_forwards_args_after_image():
    cmd = build_run_command(
        "docker",
        ["--web", "--web-port", "9000"],
        config_mounts=[],
        secret_env=[],
    )
    image_idx = cmd.index(IMAGE_TAG)
    assert cmd[image_idx + 1:] == ["--web", "--web-port", "9000"]


def test_build_run_command_includes_secret_env():
    cmd = build_run_command(
        "docker",
        [],
        config_mounts=[],
        secret_env=["-e", "OPENROUTER_API_KEY=xyz"],
    )
    i = cmd.index("-e")
    assert cmd[i + 1] == "OPENROUTER_API_KEY=xyz"


def test_build_run_command_auto_mounts_data_path(tmp_path):
    data = tmp_path / "tts"
    cmd = build_run_command(
        "docker",
        ["--data-path", str(data)],
        config_mounts=[],
        secret_env=[],
    )
    mount = f"{data}:{data}:rw"
    assert mount in cmd
    assert cmd[cmd.index(mount) - 1] == "-v"
    assert data.is_dir()  # host dir is created so docker doesn't make it root-owned


def test_build_run_command_auto_mounts_data_path_equals_form(tmp_path):
    data = tmp_path / "tts"
    cmd = build_run_command(
        "docker",
        [f"--data-path={data}"],
        config_mounts=[],
        secret_env=[],
    )
    assert f"{data}:{data}:rw" in cmd


def test_build_run_command_auto_mounts_documents_path_ro(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    cmd = build_run_command(
        "docker",
        ["--documents-path", str(docs)],
        config_mounts=[],
        secret_env=[],
    )
    assert f"{docs}:{docs}:ro" in cmd


def test_build_run_command_no_auto_mount_without_path_flags():
    cmd = build_run_command(
        "docker", ["--web"], config_mounts=[], secret_env=[]
    )
    # No extra :rw / :ro host mounts beyond the named data volume.
    assert not any(a.endswith(":rw") for a in cmd)
    assert not any(a.endswith(":ro") for a in cmd)


def test_build_run_command_sets_host_user_on_bind_mount(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("--user only applied on Linux")
    monkeypatch.setattr("os.getuid", lambda: 1234)
    monkeypatch.setattr("os.getgid", lambda: 5678)
    cmd = build_run_command(
        "docker",
        ["--data-path", str(tmp_path / "d")],
        config_mounts=[],
        secret_env=[],
    )
    i = cmd.index("--user")
    assert cmd[i + 1] == "1234:5678"


def test_build_run_command_no_host_user_without_bind_mounts(monkeypatch):
    monkeypatch.setattr("os.getuid", lambda: 1234, raising=False)
    monkeypatch.setattr("os.getgid", lambda: 5678, raising=False)
    cmd = build_run_command(
        "docker", ["--web"], config_mounts=[], secret_env=[]
    )
    assert "--user" not in cmd


def test_build_run_command_host_user_on_explicit_mount(monkeypatch):
    if sys.platform != "linux":
        pytest.skip("--user only applied on Linux")
    monkeypatch.setattr("os.getuid", lambda: 1234)
    monkeypatch.setattr("os.getgid", lambda: 5678)
    cmd = build_run_command(
        "docker", [], config_mounts=[], secret_env=[],
        mounts=["/a:/b:rw"],
    )
    assert "--user" in cmd


def test_check_docker_missing_exits(monkeypatch):
    from container_launcher import _check_docker

    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(SystemExit) as exc:
        _check_docker()
    assert exc.value.code == 1


def test_collect_secret_env_honors_host_env(monkeypatch):
    from container_launcher import _collect_secret_env

    # Clear all bridged secret env vars so only OPENROUTER_API_KEY is set.
    for _, env_name in [
        ("host_key", "OPENROUTER_API_KEY"),
        ("ollama_api_key", "OLLAMA_API_KEY"),
        ("openweathermap_api_key", "OPENWEATHERMAP_API_KEY"),
        ("telegram_bot_token", "TELEGRAM_BOT_TOKEN"),
        ("viber_bot_token", "VIBER_BOT_TOKEN"),
        ("github_token", "GITHUB_TOKEN"),
        ("huggingface_token", "HF_TOKEN"),
    ]:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")

    args = _collect_secret_env()
    assert "OPENROUTER_API_KEY=from-env" in args
    i = args.index("OPENROUTER_API_KEY=from-env")
    assert args[i - 1] == "-e"
