# OCR Tool Integration Plan

Add an open-weights OCR capability to OnIt, exposed as an MCP tool that the agent
invokes when it needs to read document images: scanned PDFs, photos of pages,
screenshots, receipts, forms.

## 1. Model choice: PaddleOCR-VL (v1.6)

| | PaddleOCR-VL 1.6 | DeepSeek-OCR | GOT-OCR 2.0 | Granite-Docling |
|---|---|---|---|---|
| License | Apache-2.0 | MIT | Apache-2.0 | Apache-2.0 |
| Params | ~0.9B | ~3B MoE (~570M active) | ~580M | 258M |
| VRAM (FP16) | ~2 GB | ~8 GB | ~3 GB | ~0.5 GB |
| OmniDocBench v1.6 | 96.33 (leader) | mid-pack | n/a | n/a |
| Languages | 100+ | many | printed/formatted | English |
| Serving | vLLM, OpenAI-compatible | vLLM/SGLang | vLLM/Ollama | vLLM/Docling |

**Pick PaddleOCR-VL** as the default:

- Top of OmniDocBench v1.6 among open models; strong on tables, formulas, mixed
  scripts, and complex layouts in a single pass.
- ~2 GB VRAM — fits on the existing GPU host alongside the Qwen chat model.
- Served by plain `vllm serve` with an OpenAI-compatible `/v1/chat/completions`
  endpoint — the exact client stack OnIt already uses (`AsyncOpenAI` in
  `src/model/serving/chat.py`), so no new inference client code.
- Apache-2.0, no commercial restrictions.

Because the integration surface is just "OpenAI-compatible endpoint + prompt",
swapping to DeepSeek-OCR (bulk throughput) or any future model is a config
change, not a code change.

### Running PaddleOCR-VL as a second vLLM instance (port 8001)

The OCR model runs in its **own vLLM instance** on the GPU host, next to the
existing chat-model instance. It is completely independent: separate process,
separate port, separate `--gpu-memory-utilization` budget.

**1. Install** — PaddleOCR-VL needs vLLM ≥ 0.11.1. Use a separate virtualenv so
it can be upgraded without touching the chat model's vLLM:

```bash
python3 -m venv ~/venvs/vllm-ocr && source ~/venvs/vllm-ocr/bin/activate
pip install -U "vllm>=0.11.1"
```

**2. Serve** (flags per the official vLLM recipe, plus port and memory cap):

```bash
vllm serve PaddlePaddle/PaddleOCR-VL \
    --port 8001 \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization <fraction giving ~6 GB, see below>
```

**3. GPU memory** — the model itself is small; the thing to control is vLLM's
preallocation:

- Weights: ~0.9B params in BF16 ≈ **2 GB**.
- Comfortable working budget (weights + activations + KV cache for single-page
  OCR requests): **~6 GB**.
- vLLM *preallocates* `gpu_memory_utilization × total VRAM` at startup
  (default 0.9). **Without an explicit `--gpu-memory-utilization`, the second
  instance will try to claim 90% of the card and OOM against the chat model.**
  Set it to the fraction that equals ~6 GB on your card:

  | Card | flag value |
  |---|---|
  | 80 GB (A100/H100) | `--gpu-memory-utilization 0.08` |
  | 48 GB (L40S/A6000) | `--gpu-memory-utilization 0.13` |
  | 24 GB (4090/L4) | `--gpu-memory-utilization 0.25` |

  If it fits, lower `--max-num-batched-tokens` (e.g. 8192) to shrink the
  budget further; it only limits batch concurrency, not accuracy.

**4. Verify** the OpenAI-compatible endpoint with the model's native prompt
format (`"OCR:"` for text; also supports `"Table Recognition:"`,
`"Formula Recognition:"`, `"Chart Recognition:"`):

```bash
curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "PaddlePaddle/PaddleOCR-VL",
  "temperature": 0,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
    {"type": "text", "text": "OCR:"}
  ]}]
}'
```

**5. Point OnIt at it**: run `onit setup` and enter
`http://<gpu-host>:8001/v1` at the "OCR endpoint URL" prompt (stored as
`serving.ocr_host` in `~/.onit/config.yaml`). `ONIT_OCR_HOST` env var and
`options.ocr_host` in the servers config also work — see §3.

For unattended operation, wrap step 2 in a systemd unit or a
`docker-compose` service on the GPU host (Baidu also publishes a prebuilt
`paddleocr-genai-vllm-server` image if a container is preferred).

Note: plain vLLM serving gives per-image text/markdown recognition, which is
what the tool needs (it sends one page image per request). The heavier
PaddleOCR "doc parser" pipeline (separate layout model + reading-order sort) is
NOT required and is out of scope.

## 2. Where the tool lives

New task module following the `tasks/local/search` pattern:

```
src/mcp/servers/tasks/vision/
    __init__.py
    ocr/
        __init__.py
        toolkit.py      # OCRToolkit: endpoint client, PDF rasterizing, batching
        mcp_server.py   # @mcp.tool ocr(...) + standalone run()
```

Re-export the tool into the consolidated **ToolsMCPServer**
(`src/mcp/servers/tasks/tools/mcp_server.py`), exactly like `local_search` /
`index_documents` are today. This guarantees discovery: `OnIt._DEFAULT_MCP_SERVERS`
(`src/onit.py`) only lists PromptsMCPServer and ToolsMCPServer.

> Observation, decide separately: `VLMToolsMCPServer` (`src/mcp/vlm_web/tools.py`,
> port 18222) runs by default but is *not* in `_DEFAULT_MCP_SERVERS`, so its
> `view_image_from_url` tool is never discovered unless the user config lists it.
> The OCR tool intentionally does NOT go there for that reason.

### Registration is conditional

Register the `ocr` tool only when an OCR endpoint is configured (mirror the
`ONIT_DISABLE_LOCAL_SEARCH` / weather-key conditional-registration patterns).
No endpoint → tool absent → agent never sees it → zero behavior change for
existing deployments.

## 3. Tool contract

```python
@mcp.tool(
    title="OCR Document Image",
    description="""Extract text from images of documents (scanned PDFs, photos
of pages, screenshots, receipts, forms). Use this when read_file returns empty
or garbled text for a PDF, or when the file is an image (png/jpg/webp/tiff/bmp)
containing text.

Args:
- path: Image or PDF file path within data_path folder (required)
- pages: For PDFs — page range to OCR, e.g. "1-5" (default: "1-10")
- output: "markdown" (default; preserves tables/headings) or "text"
- max_chars: Truncate combined output (default: 100000)

Returns JSON: {content, path, pages_processed, format, status}"""
)
def ocr(path=None, pages="1-10", output="markdown", max_chars=100000) -> str: ...
```

Behavior:

1. Validate + jail `path` under `DATA_PATH` with the existing helpers (the MCP
   sandbox path-jail applies here like everywhere else).
2. Image file → one request. PDF → rasterize the requested pages at ~200 DPI
   with **pypdfium2** (new dependency; `pypdf` cannot rasterize), one request
   per page, concatenate with `--- page N ---` markers. Cap pages per call
   (default 10) so one tool call can't monopolize the endpoint.
3. Each request: base64 data-URL image + the model's native task prompt
   (`"OCR:"`; tables/formulas come back as markdown/LaTeX) via
   `openai.OpenAI(base_url=OCR_HOST).chat.completions.create(...)`,
   `temperature=0`, request timeout ~120 s.
4. Errors return `{"error": ..., "status": "error"}` JSON like every other tool.

### Configuration via `onit setup`

The OCR endpoint is configured the same way as the LLM serving host, through
the interactive wizard (`src/setup.py`):

1. **`src/setup.py`** — add to `SETTINGS`:

   ```python
   ("serving.ocr_host", "OCR endpoint URL (blank = OCR tool disabled)", ""),
   ```

   The wizard then prompts for it alongside "LLM endpoint URL", persists it to
   `~/.onit/config.yaml` under `serving.ocr_host`, and `onit setup --show`
   displays it automatically (both iterate `SETTINGS`).

2. **`src/cli.py`** — before `_ensure_mcp_servers()` spawns the server
   processes, resolve and inject the env var, mirroring the weather-key
   pattern at `cli.py:818-824`:

   ```python
   ocr_host = (config's serving.ocr_host) or os.environ.get('ONIT_OCR_HOST')
   if ocr_host:
       os.environ['ONIT_OCR_HOST'] = ocr_host
   ```

   The spawned ToolsMCPServer inherits the env var. When unset, the `ocr`
   tool is not registered and `_setup_servers()` prints an availability note
   like the weather/web-search warnings:
   `"OCR endpoint not configured. OCR tool is unavailable. Run: onit setup"`.

3. **Server-side resolution** (in the tools server `run()`, mirroring
   `data_path` handling):

   | setting | server-config `options.` | env fallback | default |
   |---|---|---|---|
   | endpoint | `ocr_host` | `ONIT_OCR_HOST` | unset → tool disabled |
   | model id | `ocr_model` | `ONIT_OCR_MODEL` | auto-resolve first model from `/v1/models` |
   | api key | `ocr_api_key` | `ONIT_OCR_API_KEY` | `EMPTY` |

   Model id and API key stay out of the wizard to keep it short — a dedicated
   vLLM instance serves exactly one model (auto-resolved from `/v1/models`)
   and typically needs no key; both remain overridable via env/config for
   non-default deployments.

## 4. Steering the agent to the tool

- The tool description above carries the routing hint ("use when read_file
  returns empty/garbled text for a PDF, or the file is an image").
- `read_file` mode="text": when a PDF extracts to (near-)empty text, add
  `"note": "PDF has no text layer (likely scanned); use the ocr tool"` to the
  returned JSON so the model self-corrects mid-task without prompt changes.
- Mention the tool in the ToolsMCPServer description string in
  `configs/default.yaml` and `_DEFAULT_MCP_SERVERS`.

## 5. Testing

- **Unit** (`src/test/test_ocr_tool.py`, mocked OpenAI client):
  path jail rejection, image vs PDF dispatch, page-range parsing and cap,
  truncation, markdown/text prompts, endpoint-unset ⇒ tool not registered,
  endpoint error ⇒ error JSON.
- **Fixture**: tiny scanned (image-only) PDF + a PNG of text under `src/test/data/`.
- **Live e2e** (this machine can reach the GPU host; same pattern as existing
  live chat tests): deploy the model, set `ONIT_OCR_HOST`, run `onit`, ask
  "what does ~/sandbox/scan.pdf say?" and verify the agent calls `ocr` and
  answers from its output. Gate behind an env flag like other live tests.
- `test_tool_discovery.py` / `test_tool_registry.py`: assert `ocr` appears
  when configured.

## 6. Docs

- README tool table: add `ocr`.
- `docs/OCR_SERVING.md`: vLLM serve command, VRAM budget, config/env settings,
  model-swap instructions (DeepSeek-OCR alternative).

## 7. Phases

1. **Toolkit + tool + unit tests** — mocked endpoint; conditional registration;
   `pypdfium2` added to `pyproject.toml` dependencies; `onit setup` wiring
   (`serving.ocr_host` in `setup.py` SETTINGS, env injection + availability
   warning in `cli.py`).
2. **Deploy + e2e** — `vllm serve PaddlePaddle/PaddleOCR-VL` on the GPU host,
   live end-to-end verification through the chat loop and web UI (image upload
   → `/uploads/{sid}/` → session data_path → ocr tool).
3. **Polish** — scanned-PDF hint in `read_file`, docs, README.

## Open questions

- Should `ocr` also accept URLs, or is local-path-only (after upload/fetch)
  enough? Local-only keeps the jail semantics simple; `fetch_content` +
  upload flow already lands files in data_path.
- Whether to fold OCR into `read_file` as `mode="ocr"` later once the
  standalone tool proves out (keeps the tool count down).
