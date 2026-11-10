"""Start the app:  python -m bs2autotune"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bs2autotune",
                                     description="Automatic sound matching for the Novation Bass Station II.")
    parser.add_argument("--port", type=int, default=8765, help="port of the local web interface (default 8765)")
    parser.add_argument("--data", default=None, help="folder for settings, calibration and results")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--demo", action="store_true", help="start with the built-in virtual synth (no hardware)")
    parser.add_argument("--list-devices", action="store_true", help="print the MIDI and audio devices and exit")
    args = parser.parse_args(argv)

    if args.list_devices:
        from .hardware import list_audio_inputs, list_midi_ports

        midi = list_midi_ports()
        audio = list_audio_inputs()
        print("MIDI outputs:")
        for name in midi["outputs"]:
            print("  ", name)
        print("MIDI inputs:")
        for name in midi["inputs"]:
            print("  ", name)
        if midi["error"]:
            print("  (MIDI problem: %s)" % midi["error"])
        print("Audio inputs:")
        for dev in audio["devices"]:
            print("   [%d] %s (%d ch, %d Hz)%s" % (dev["index"], dev["name"], dev["channels"], dev["samplerate"],
                                                    " *" if dev["default"] else ""))
        if audio["error"]:
            print("  (audio problem: %s)" % audio["error"])
        return 0

    try:
        import numba  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        print("A required package is missing: %s\nInstall the requirements first:  pip install -r requirements.txt"
              % exc, file=sys.stderr)
        return 1

    from .app import App
    from .server import make_server, warm_up

    data_dir = args.data or os.environ.get("BS2AUTOTUNE_DATA") or os.path.join(os.getcwd(), "data")
    app = App(data_dir)
    if args.demo:
        app.update_settings({"mode": "demo"})
    server = make_server(app, args.port)
    url = "http://127.0.0.1:%d/" % server.server_address[1]
    print("BS2 Autotune is running at %s  (Ctrl+C to quit)" % url)
    print("Data folder: %s" % app.data_dir)
    warm_up(app)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ...")
        app.stop_job()
        app.wait(5.0)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
