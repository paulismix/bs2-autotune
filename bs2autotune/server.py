"""Local web server for the user interface.

Standard library only.  The server listens on 127.0.0.1 and additionally
checks the Host and Origin headers, so no other machine and no other web site
open in the browser can talk to it.  It exposes a small JSON API, a stream of
server-sent events for live progress and the files of the single-page UI.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .analysis import AudioError
from .app import App, Busy, UserError
from .device import DeviceError
from .params import describe_patch, init_patch, normalize_patch, patch_text

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css",
          "/favicon.svg": "favicon.svg"}
MAX_UPLOAD = 400 * 1024 * 1024
MEDIA_REFERENCE = {"excerpt.wav", "excerpt.png", "audition.wav"}
MEDIA_SESSION = {"reference.wav", "synth.wav", "reference.png", "synth.png", "patch.txt", "patch.syx", "result.json"}
CSP = ("default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; "
       "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "BS2Autotune"
    app: App = None  # type: ignore[assignment]

    # -- plumbing ---------------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # keep the terminal quiet
        return

    def _allowed_hosts(self) -> Tuple[str, ...]:
        port = self.server.server_address[1]
        return (f"127.0.0.1:{port}", f"localhost:{port}")

    def _check_origin(self, write: bool) -> bool:
        """Only this page, opened from this machine, may use the API."""
        if self.headers.get("Host", "") not in self._allowed_hosts():
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in tuple(f"http://{h}" for h in self._allowed_hosts()):
            return False
        if write and origin is None and self.headers.get("Sec-Fetch-Site", "same-origin") not in (
                "same-origin", "none"):
            return False
        return True

    def _send(self, status: int, body: bytes, content_type: str, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self, limit: int = 2 * 1024 * 1024) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length < 0 or length > limit:
            raise UserError("The request is too large.")
        data = b""
        while len(data) < length:
            chunk = self.rfile.read(min(1 << 20, length - len(data)))
            if not chunk:
                break
            data += chunk
        return data

    def _json_body(self) -> dict:
        if "application/json" not in self.headers.get("Content-Type", ""):
            raise UserError("Expected JSON.")
        raw = self._body()
        data = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(data, dict):
            raise UserError("Expected a JSON object.")
        return data

    def _file(self, path: str, download: Optional[str] = None) -> None:
        if not os.path.isfile(path):
            self._error(404, "Not found.")
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".js"):
            ctype = "text/javascript; charset=utf-8"
        elif path.endswith((".html", ".css", ".txt", ".json", ".svg")):
            ctype = {".html": "text/html", ".css": "text/css", ".txt": "text/plain", ".json": "application/json",
                     ".svg": "image/svg+xml"}[os.path.splitext(path)[1]] + "; charset=utf-8"
        size = os.path.getsize(path)
        start, end = 0, size - 1
        status = 200
        rng = self.headers.get("Range")
        if rng:  # Safari insists on range requests for audio
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        extra = {"Accept-Ranges": "bytes"}
        if status == 206:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        if download:
            extra["Content-Disposition"] = f'attachment; filename="{download}"'
        self._send(status, body, ctype, extra)

    # -- routes ---------------------------------------------------------------
    def do_GET(self) -> None:
        if not self._check_origin(False):
            self._error(403, "Forbidden.")
            return
        url = urlparse(self.path)
        path = url.path
        app = self.app
        try:
            if path in STATIC:
                self._file(os.path.join(WEB_DIR, STATIC[path]))
            elif path == "/api/state":
                self._json(app.state())
            elif path == "/api/devices":
                self._json(app.list_devices())
            elif path == "/api/events":
                self._events()
            elif path == "/api/init-patch":
                self._json({"rows": describe_patch(init_patch())})
            elif path.startswith("/api/library/"):
                self._json(app.library_get(path.rsplit("/", 1)[1]))
            elif path.startswith("/media/reference/"):
                name = path.rsplit("/", 1)[1]
                if name not in MEDIA_REFERENCE:
                    raise UserError("Not found.")
                self._file(app.path("reference", name))
            elif path.startswith("/media/session/"):
                m = re.fullmatch(r"/media/session/(\d{8}-\d{6})/([a-z.]+)", path)
                if not m or m.group(2) not in MEDIA_SESSION:
                    raise UserError("Not found.")
                download = None
                if m.group(2) in ("patch.txt", "patch.syx", "result.json"):
                    download = f"bs2-autotune-{m.group(1)}" + os.path.splitext(m.group(2))[1]
                self._file(app.path("sessions", m.group(1), m.group(2)), download)
            else:
                self._error(404, "Not found.")
        except UserError as exc:
            self._error(404, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_HEAD = do_GET

    def do_POST(self) -> None:
        if not self._check_origin(True):
            self._error(403, "Forbidden.")
            return
        url = urlparse(self.path)
        path = url.path
        app = self.app
        try:
            if path == "/api/reference":
                if self.headers.get("X-BS2-Upload") != "1":
                    raise UserError("Bad upload.")
                name = parse_qs(url.query).get("name", ["reference"])[0]
                payload = self._body(MAX_UPLOAD)
                self._json(app.set_reference_file(name, payload))
                return
            data = self._json_body()
            if path == "/api/settings":
                self._json(app.update_settings(data))
            elif path == "/api/signal":
                self._json(app.job_test_signal().to_dict())
            elif path == "/api/calibrate":
                self._json(app.job_calibrate().to_dict())
            elif path == "/api/analyse":
                self._json(app.job_analyse(data).to_dict())
            elif path == "/api/match":
                self._json(app.job_match(data).to_dict())
            elif path == "/api/stop":
                app.stop_job()
                self._json({"ok": True})
            elif path == "/api/patch/send":
                patch = data.get("patch")
                if patch is None and data.get("library"):
                    patch = app.library_get(str(data["library"]))["patch"]
                if patch is None and app.result is not None:
                    patch = app.result["patch"]
                if patch is None:
                    raise UserError("There is no patch to send yet.")
                self._json(app.job_send_patch(normalize_patch(patch), bool(data.get("play"))).to_dict())
            elif path == "/api/patch/text":
                patch = normalize_patch(data.get("patch") or {})
                self._json({"text": patch_text(patch, str(data.get("title", "BS2 patch"))[:60])})
            elif path == "/api/library/save":
                self._json(app.library_save(str(data.get("name", "")), data.get("patch")))
            elif path == "/api/library/delete":
                app.library_delete(str(data.get("id", "")))
                self._json({"ok": True})
            else:
                self._error(404, "Not found.")
        except Busy as exc:
            self._error(409, str(exc))
        except (UserError, DeviceError, AudioError) as exc:
            self._error(400, str(exc))
        except json.JSONDecodeError:
            self._error(400, "Malformed JSON.")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # last resort: report instead of dropping the connection
            self._error(500, f"{exc.__class__.__name__}: {exc}")

    def _events(self) -> None:
        """Server-sent events: job progress and 'something changed' notifications."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        q = self.app.bus.subscribe()
        try:
            self.wfile.write(b"retry: 1500\n\n")
            self.wfile.flush()
            while True:
                try:
                    event = q.get(timeout=12.0)
                    payload = f"data: {json.dumps(event)}\n\n".encode("utf-8")
                except queue.Empty:
                    payload = b": keep-alive\n\n"
                self.wfile.write(payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.bus.unsubscribe(q)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(app: App, port: int = 8765) -> Server:
    handler = type("BoundHandler", (Handler,), {"app": app})
    last_error: Optional[Exception] = None
    for candidate in ([port] if port == 0 else range(port, port + 20)):
        try:
            return Server(("127.0.0.1", candidate), handler)
        except OSError as exc:  # port taken: try the next one
            last_error = exc
    raise OSError(f"No free port near {port}: {last_error}")


def warm_up(app: App) -> None:
    """Compile the numeric kernels in the background (slow only on the very first start)."""
    def work() -> None:
        try:
            from . import engine, features
            t0 = time.time()
            engine.warmup()
            features.warmup()
            app.bus.publish({"type": "ready", "seconds": round(time.time() - t0, 1)})
        except Exception:
            pass

    threading.Thread(target=work, name="bs2warmup", daemon=True).start()
