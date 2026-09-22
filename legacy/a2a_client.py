"""The ``onit ask`` client, moved beside the A2A server it speaks to.

The ``ask`` subcommand was removed from the active CLI when its only bundled
server, ``legacy/a2a_server``, was extracted (see legacy/README.md). The
client itself is a plain JSON-RPC-over-HTTP implementation with no A2A SDK
import, so it lives here and is driven directly:

    python -m legacy.a2a_client "what is the weather in Manila" \
        --server http://localhost:9001

It talks to any A2A server, not just the bundled one.
"""

import base64
import json
import os
import sys
import threading
import time

import requests


def download_files(text: str, server_url: str) -> str:
    """Download any files referenced in the response text from the A2A server."""
    import re
    # Match /uploads/filename patterns in the text
    pattern = re.compile(r'/uploads/([^\s\)\]"\'<>`*]+)')
    downloaded = []
    for match in pattern.finditer(text):
        filename = match.group(1)
        download_url = f"{server_url.rstrip('/')}/uploads/{filename}"
        try:
            resp = requests.get(download_url, timeout=60)
            resp.raise_for_status()
            local_path = os.path.join(os.getcwd(), os.path.basename(filename))
            with open(local_path, "wb") as f:
                f.write(resp.content)
            downloaded.append(local_path)
        except Exception as e:
            downloaded.append(f"Failed to download {filename}: {e}")
    if downloaded:
        text += "\n\nDownloaded files:\n" + "\n".join(f"  - {p}" for p in downloaded)
    return text


def upload_file(url: str, filepath: str) -> str:
    """Upload a file to the A2A server and return the uploaded filename."""
    filepath = os.path.abspath(os.path.expanduser(filepath))
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        resp = requests.post(
            f"{url.rstrip('/')}/uploads/",
            files={'file': (filename, f)},
            timeout=60,
        )
        resp.raise_for_status()
    return filename


def build_a2a_parts(task: str, file: str = None, image: str = None) -> list:
    """Build the A2A message parts list from task text and optional files."""
    import mimetypes as _mimetypes

    parts = [{"kind": "text", "text": task}]

    if file:
        filepath = os.path.abspath(os.path.expanduser(file))
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        mime_type = _mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
        with open(filepath, 'rb') as f:
            file_data = base64.b64encode(f.read()).decode('utf-8')
        parts.append({
            "kind": "file",
            "file": {
                "bytes": file_data,
                "mimeType": mime_type,
                "name": os.path.basename(filepath),
            }
        })

    if image:
        mime_types = {
            '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp',
            '.tiff': 'image/tiff', '.tif': 'image/tiff',
        }
        ext = os.path.splitext(image)[1].lower()
        mime_type = mime_types.get(ext, 'image/png')
        with open(image, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        parts.append({
            "kind": "file",
            "file": {
                "bytes": image_data,
                "mimeType": mime_type,
                "name": os.path.basename(image),
            }
        })

    return parts


def extract_a2a_text(result: dict) -> str | None:
    """Extract text from an A2A result dict (Task or Message)."""
    text = None
    if "status" in result:
        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    text = part["text"]
                    break
            if text:
                break
        if not text:
            task_result = result.get("result")
            if task_result:
                for part in task_result.get("parts", []):
                    if part.get("kind") == "text":
                        text = part["text"]
                        break
    if not text and "parts" in result:
        for part in result.get("parts", []):
            if part.get("kind") == "text":
                text = part["text"]
                break
    return text


class StreamState:
    """Mutable state shared between SSE streaming helpers."""

    def __init__(self, stop_timer: threading.Event, timer_thread: threading.Thread):
        self.stop_timer = stop_timer
        self.timer_thread = timer_thread
        self.printed_len: int = 0
        self.final_text: str | None = None
        self.raw_result: dict = {}
        self.spinner_cleared: bool = False
        self.cursor_shown: bool = False

    def erase_cursor(self) -> None:
        """Remove the blinking block cursor and restore the terminal cursor."""
        if self.cursor_shown:
            sys.stdout.write("\b \b")
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            self.cursor_shown = False

    def show_cursor(self) -> None:
        """Hide terminal cursor and show a blinking white block instead."""
        if not self.cursor_shown:
            sys.stdout.write("\033[?25l")
            sys.stdout.write("\033[5m█\033[0m")
            sys.stdout.flush()
            self.cursor_shown = True

    def clear_spinner(self) -> None:
        """Stop the elapsed-time spinner and clear its line."""
        if not self.spinner_cleared:
            self.stop_timer.set()
            self.timer_thread.join()
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            self.spinner_cleared = True


def handle_sse_events(resp: requests.Response, state: StreamState) -> None:
    """Process SSE event lines from a streaming A2A response.

    Updates *state* in place with streamed text deltas, the final text,
    and the raw result dict.
    """
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue

        result = event.get("result", {})
        status = result.get("status", {})
        event_state = status.get("state", "")

        if event_state == "working":
            msg = status.get("message", {})
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    full = part["text"]
                    if len(full) > state.printed_len:
                        state.clear_spinner()
                        state.erase_cursor()
                        sys.stdout.write(full[state.printed_len:])
                        sys.stdout.flush()
                        state.printed_len = len(full)
                        state.show_cursor()
                    break

        elif event_state == "completed":
            state.raw_result = result
            state.final_text = extract_a2a_text(result)
            if not state.final_text:
                msg = status.get("message", {})
                for part in msg.get("parts", []):
                    if part.get("kind") == "text":
                        state.final_text = part["text"]
                        break

        elif "parts" in result:
            state.raw_result = result
            state.final_text = extract_a2a_text(result)


def format_output(state: StreamState, url: str) -> str:
    """Produce the final return value after streaming/response is complete.

    Handles the JSON-dump fallback, trailing-text flush for streamed
    responses, and file downloads.
    """
    if state.final_text is None:
        return json.dumps(state.raw_result, indent=2)

    if state.printed_len > 0:
        remaining = state.final_text[state.printed_len:]
        if remaining:
            sys.stdout.write(remaining)
        sys.stdout.write("\n")
        sys.stdout.flush()

    if "/uploads/" in state.final_text:
        state.final_text = download_files(state.final_text, url)

    if state.printed_len > 0:
        return ""

    return state.final_text


def send_task(url: str, task: str, file: str = None, image: str = None) -> str:
    """Send a task to an OnIt A2A server using SSE streaming.

    Uses ``message/stream`` so the server can push incremental
    ``TaskStatusUpdateEvent`` (state=working) events.  Each event
    carries the accumulated text so far; the client prints only the
    new delta.  The "Waiting ..." spinner is replaced by live output
    as soon as the first token arrives.

    Falls back to the non-streaming ``message/send`` path if the SSE
    request fails (e.g. older server without streaming support).
    """
    parts = build_a2a_parts(task, file=file, image=image)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": parts,
                "messageId": "client-001",
            }
        },
    }

    # Elapsed time indicator while waiting for response
    stop_timer = threading.Event()
    start = time.monotonic()

    def _show_elapsed():
        while not stop_timer.is_set():
            elapsed = int(time.monotonic() - start)
            h, remainder = divmod(elapsed, 3600)
            m, s = divmod(remainder, 60)
            sys.stderr.write(f"\rWaiting... {h:02d}:{m:02d}:{s:02d}")
            sys.stderr.flush()
            stop_timer.wait(1.0)

    sys.stderr.write("\rWaiting... 00:00:00")
    sys.stderr.flush()

    timer_thread = threading.Thread(target=_show_elapsed, daemon=True)
    timer_thread.start()

    state = StreamState(stop_timer, timer_thread)

    try:
        resp = requests.post(
            url.rstrip("/"),
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=None,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            handle_sse_events(resp, state)
        else:
            # Non-streaming JSON response (fallback)
            data = resp.json()
            error = data.get("error")
            if error:
                state.clear_spinner()
                return f"Error: {error}"
            state.raw_result = data.get("result", {})
            state.final_text = extract_a2a_text(state.raw_result)

    except requests.RequestException:
        # SSE failed — fall back to non-streaming message/send
        state.clear_spinner()
        payload["method"] = "message/send"
        resp = requests.post(url.rstrip("/"), json=payload, timeout=None)
        resp.raise_for_status()
        data = resp.json()
        error = data.get("error")
        if error:
            return f"Error: {error}"
        state.raw_result = data.get("result", {})
        state.final_text = extract_a2a_text(state.raw_result)
    finally:
        state.erase_cursor()
        state.clear_spinner()

    return format_output(state, url)


def _main() -> None:
    """CLI: python -m legacy.a2a_client "task" --server URL"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Send a task to an OnIt A2A server and print the response.")
    parser.add_argument("task", type=str, help="Task to send to the A2A server.")
    parser.add_argument("--file", type=str, default=None,
                        help="File to upload along with the task.")
    parser.add_argument("--image", type=str, default=None,
                        help="Image file for vision processing (model must be a VLM).")
    parser.add_argument("--server", type=str, default="http://localhost:9001",
                        help="A2A server URL (default: http://localhost:9001).")
    args = parser.parse_args()

    if args.image:
        valid_image_ext = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp',
                           '.tiff', '.tif'}
        image_path = os.path.abspath(os.path.expanduser(args.image))
        if not os.path.isfile(image_path):
            print(f"Error: Image file not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in valid_image_ext:
            print(f"Error: Invalid image file. Supported formats: "
                  f"{', '.join(sorted(valid_image_ext))}", file=sys.stderr)
            sys.exit(1)
        args.image = image_path

    try:
        answer = send_task(args.server, args.task, file=args.file, image=args.image)
        print(answer)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
