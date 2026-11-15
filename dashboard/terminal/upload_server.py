#!/usr/bin/python3
"""upload_server.py — receives a photo from the terminal page and saves it.

The owner taps "photo" in the terminal; the page posts the picture to
/term/upload; this writes it under /data/jack-uploads and answers with the
path, which the page then types into the chat so Claude can open it.

Started by systemd only when a photo arrives (jack-upload.socket), and exits
after two idle minutes: nothing runs between photos. Reachable only from the
web server's network. It accepts a request only if it carries the terminal's
own long password (the web server forwards the sign-in cookie as a header),
only pictures, only up to 30 MB, and it chooses the file name itself.

Storage: pictures older than 30 days are deleted, and the folder is held to
500 MB — oldest first.
"""
import base64, hmac, json, os, secrets, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DIR = Path(os.environ.get("JACK_UPLOAD_DIR", "/data/jack-uploads"))
MAX_BYTES = 30 * 2**20
KEEP_DAYS, CAP_BYTES = 30, 500 * 2**20
IDLE_EXIT_S = 120
PASSWORD = os.environ.get("JACK_TERM_PASSWORD", "")
EXPECT = "Basic " + base64.b64encode(("jack:" + PASSWORD).encode()).decode()
last = [time.time()]


def kind(head: bytes):
    """The picture's real type, from its first bytes — never from what the sender says."""
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1", b"heif", b"hevc"):
        return "heic"
    return None


def tidy():
    files = sorted((f for f in DIR.rglob("*") if f.is_file()), key=lambda f: f.stat().st_mtime)
    now, total = time.time(), 0
    for f in list(files):
        if now - f.stat().st_mtime > KEEP_DAYS * 86400:
            f.unlink(missing_ok=True)
            files.remove(f)
    total = sum(f.stat().st_size for f in files)
    while files and total > CAP_BYTES:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)
    for d in DIR.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


class H(BaseHTTPRequestHandler):
    server_version = "jack-upload"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):       # no request log: nothing about the owner's photos is recorded
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        last[0] = time.time()
        if self.path.split("?")[0] != "/term/upload":
            return self.reply(404, {"error": "not here"})
        if not PASSWORD or not hmac.compare_digest(self.headers.get("Authorization", ""), EXPECT):
            return self.reply(401, {"error": "sign in to the terminal first"})
        origin, host = self.headers.get("Origin"), self.headers.get("Host", "")
        if origin and origin.split("://", 1)[-1] != host:
            return self.reply(403, {"error": "wrong site"})
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self.reply(411, {"error": "no length"})
        if n <= 0 or n > MAX_BYTES:
            return self.reply(413, {"error": "too large: the limit is 30 MB"})
        data = self.rfile.read(n)
        ext = kind(data[:16])
        if len(data) != n or ext is None:
            return self.reply(415, {"error": "that is not a picture this can take"})
        day = DIR / time.strftime("%Y-%m-%d", time.gmtime())
        day.mkdir(parents=True, exist_ok=True)
        path = day / f"photo-{time.strftime('%H%M%S', time.gmtime())}-{secrets.token_hex(3)}.{ext}"
        with open(path, "xb") as f:
            f.write(data)
        os.chmod(path, 0o600)
        try:
            tidy()
        except OSError:
            pass
        self.reply(200, {"path": str(path), "bytes": n})

    def do_GET(self):
        self.reply(405, {"error": "post a picture"})


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DIR, 0o700)
    if os.environ.get("LISTEN_FDS") == "1" and os.environ.get("LISTEN_PID") == str(os.getpid()):
        srv = HTTPServer(("0.0.0.0", 0), H, bind_and_activate=False)    # the socket systemd is holding open
        srv.socket.close()
        srv.socket = socket.socket(fileno=3)
        srv.server_address = srv.socket.getsockname()
    else:                                                               # run by hand, for a test
        srv = HTTPServer((sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 7682), H)

    def watchdog():
        while time.time() - last[0] < IDLE_EXIT_S:
            time.sleep(5)
        os._exit(0)

    threading.Thread(target=watchdog, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
