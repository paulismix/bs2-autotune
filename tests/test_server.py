"""The local web server: API, live events and its protection against other origins."""

import http.client
import json
import threading
import time

import numpy as np
import pytest

from bs2autotune.app import App
from bs2autotune.audio_util import write_wav
from bs2autotune.device import VirtualDevice
from bs2autotune.params import normalize_patch
from bs2autotune.server import make_server


@pytest.fixture()
def server(tmp_path):
    app = App(str(tmp_path / "data"))
    app.update_settings({"mode": "demo"})
    srv = make_server(app, 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield app, srv.server_address[1]
    app.stop_job()
    app.wait(10)
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    res = conn.getresponse()
    payload = res.read()
    conn.close()
    return res.status, res.getheaders(), payload


def test_pages_and_state(server):
    app, port = server
    for path, kind in (("/", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css")):
        status, headers, body = call(port, "GET", path)
        assert status == 200 and len(body) > 500
        assert dict(headers)["Content-Type"].startswith(kind)
        assert "default-src 'self'" in dict(headers)["Content-Security-Policy"]
    status, _, body = call(port, "GET", "/api/state")
    state = json.loads(body)
    assert status == 200 and state["mode"] == "demo" and state["reference"] is None
    assert set(state["efforts"]) == {"quick", "normal", "thorough"}
    assert call(port, "GET", "/api/devices")[0] == 200


def test_only_this_origin_may_use_the_api(server):
    app, port = server
    assert call(port, "GET", "/api/state", headers={"Host": "evil.example"})[0] == 403  # DNS rebinding
    assert call(port, "GET", "/api/state", headers={"Host": f"127.0.0.1:{port + 1}"})[0] == 403
    assert call(port, "POST", "/api/stop", body={}, headers={"Origin": "http://evil.example"})[0] == 403
    assert call(port, "POST", "/api/stop", body={}, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert call(port, "POST", "/api/stop", body={}, headers={"Origin": f"http://127.0.0.1:{port}"})[0] == 200
    # form posts (which browsers send across origins without asking) are not accepted
    assert call(port, "POST", "/api/settings", raw=b"mode=hardware",
                headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 400
    assert call(port, "POST", "/api/reference?name=x.wav", raw=b"RIFF")[0] == 400  # upload without the marker


def test_no_file_outside_the_whitelist_is_served(server):
    app, port = server
    for path in ("/../README.md", "/app.py", "/media/reference/../../settings.json", "/media/session/x/patch.txt",
                 "/media/session/20240101-000000/../../../settings.json", "/media/reference/secret.wav",
                 "/api/library/..%2F..%2Fsettings"):
        assert call(port, "GET", path)[0] == 404, path
    assert call(port, "POST", "/api/nothing", body={})[0] == 404


def test_settings_are_validated(server):
    app, port = server
    status, _, body = call(port, "POST", "/api/settings", body={"midi_channel": 99, "mode": "nonsense", "effort": "quick",
                                                              "unknown": 1})
    settings = json.loads(body)
    assert status == 200 and settings["midi_channel"] == 16 and settings["mode"] == "demo"
    assert settings["effort"] == "quick" and "unknown" not in settings
    assert call(port, "POST", "/api/settings", raw=b"{not json", headers={"Content-Type": "application/json"})[0] == 400


def test_upload_analyse_and_events(server, line, tmp_path):
    app, port = server
    dev = VirtualDevice(seed=11)
    dev.latency = dev.true_latency
    y = dev.capture(normalize_patch(dict(filter_freq=150, filter_env=90, mod_decay=60, mod_sustain=30, amp_release=25)), line)
    path = tmp_path / "line.wav"
    write_wav(str(path), y / np.abs(y).max() * 0.7, 32000)
    status, _, body = call(port, "POST", "/api/reference?name=my%20line.wav", raw=path.read_bytes(),
                           headers={"X-BS2-Upload": "1", "Content-Type": "application/octet-stream"})
    info = json.loads(body)
    assert status == 200 and info["filename"] == "my line.wav" and abs(info["duration"] - len(y) / 32000) < 0.01
    assert call(port, "POST", "/api/reference?name=bad.wav", raw=b"not audio at all",
                headers={"X-BS2-Upload": "1"})[0] == 400

    # live events arrive while the analysis runs
    events = []
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", "/api/events")
    stream = conn.getresponse()
    assert stream.status == 200 and stream.getheader("Content-Type").startswith("text/event-stream")

    def listen():
        try:
            while True:
                line_ = stream.fp.readline()
                if not line_:
                    break
                if line_.startswith(b"data: "):
                    events.append(json.loads(line_[6:]))
        except Exception:
            pass

    threading.Thread(target=listen, daemon=True).start()
    # the upload replaced the reference: load the good file again, then analyse a selected part
    call(port, "POST", "/api/reference?name=line.wav", raw=path.read_bytes(), headers={"X-BS2-Upload": "1"})
    status, _, body = call(port, "POST", "/api/analyse", body={"mode": "stem", "region": [2.0, 7.0]})
    assert status == 200 and json.loads(body)["kind"] == "analysis"
    app.wait(60)
    time.sleep(0.3)
    conn.close()
    state = json.loads(call(port, "GET", "/api/state")[2])
    analysis = state["reference"]["analysis"]
    assert state["job"]["status"] == "done", state["job"]
    # the excerpt is the selection (it may start a few milliseconds earlier to include the first attack)
    assert 1.9 <= analysis["excerpt"][0] < analysis["excerpt"][1] <= 7.0
    assert 6 <= len(analysis["phrase"]) <= 20 and analysis["mode"] == "stem"
    assert any(e.get("type") == "job" for e in events)
    assert call(port, "GET", "/media/reference/excerpt.wav")[0] == 200
    status, headers, body = call(port, "GET", "/media/reference/excerpt.wav", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(body) == 100 and dict(headers)["Content-Range"].startswith("bytes 0-99/")
    png = call(port, "GET", "/media/reference/excerpt.png")[2]
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_one_job_at_a_time(server):
    app, port = server
    release = threading.Event()
    app.start_job("test", "Slow job", lambda job: release.wait(10))
    try:
        status, _, body = call(port, "POST", "/api/calibrate", body={})
        assert status == 409 and "still running" in json.loads(body)["error"]
    finally:
        release.set()
        app.wait(10)
    assert app.job.status == "done"
