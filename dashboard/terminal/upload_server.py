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
from urllib.parse import parse_qs
import subprocess

DIR = Path(os.environ.get("JACK_UPLOAD_DIR", "/data/jack-uploads"))
MAX_BYTES = 30 * 2**20
KEEP_DAYS, CAP_BYTES = 30, 500 * 2**20
IDLE_EXIT_S = 120
PASSWORD = os.environ.get("JACK_TERM_PASSWORD", "")
EXPECT = "Basic " + base64.b64encode(("jack:" + PASSWORD).encode()).decode()
last = [time.time()]


TMUX_SOCKET = os.environ.get("JACK_TMUX_SOCKET", "jack")      # the one shared session the page shows
TMUX_SESSION = os.environ.get("JACK_TMUX_SESSION", "jack")
MAX_HISTORY = 5000


def history(lines: int) -> str:
    """The current window as plain text, newest at the bottom: what has scrolled
    away plus what is on screen. The page's copy view shows it, because a phone
    cannot select text inside a terminal and the terminal itself holds only one
    screenful. Nothing is stored: tmux is asked each time."""
    n = max(0, min(int(lines), MAX_HISTORY))
    out = subprocess.run(["/usr/bin/tmux", "-L", TMUX_SOCKET, "capture-pane", "-p", "-J",
                          "-t", TMUX_SESSION + ":", "-S", f"-{n}"],
                         capture_output=True, timeout=5)
    if out.returncode != 0:
        raise RuntimeError("tmux refused")
    return out.stdout.decode("utf-8", "replace")


def tmux(*args, timeout=5):
    return subprocess.run(["/usr/bin/tmux", "-L", TMUX_SOCKET, *args], capture_output=True, timeout=timeout)


def tmux_act(q):
    """Run a tmux control action server-side (rename a window, open a new one,
    switch). The page calls this instead of typing tmux's own command prompt
    over the wire — a phone's network can collapse those keystrokes and spill
    the command into whatever is focused. Here there is no timing and no spill.
    The window name is passed as a lone argument (no shell), index as digits."""
    action = (q.get("action") or [""])[0]
    sess = TMUX_SESSION
    if action == "rename":
        win = (q.get("win") or [""])[0]
        if not win.isdigit():
            return 400, {"error": "bad window"}
        name = "".join(c for c in (q.get("name") or [""])[0] if c.isprintable()).strip()[:32]
        if not name:
            return 400, {"error": "empty name"}
        tmux("set-window-option", "-t", f"{sess}:{win}", "automatic-rename", "off")
        r = tmux("rename-window", "-t", f"{sess}:{win}", "--", name)
        return (200, {"ok": True}) if r.returncode == 0 else (503, {"error": "rename failed"})
    if action == "newwin":
        where = (q.get("where") or ["server"])[0]
        if where == "laptop":
            r = tmux("new-window", "-t", f"{sess}:", "-n", "Laptop", "exec ssh -tt laptop")
            tmux("set-window-option", "-t", f"{sess}:", "automatic-rename", "off")
        else:
            r = tmux("new-window", "-t", f"{sess}:")
        return (200, {"ok": True}) if r.returncode == 0 else (503, {"error": "new window failed"})
    if action == "select":
        win = (q.get("win") or [""])[0]
        if not win.isdigit():
            return 400, {"error": "bad window"}
        r = tmux("select-window", "-t", f"{sess}:{win}")
        return (200, {"ok": True}) if r.returncode == 0 else (503, {"error": "switch failed"})
    return 400, {"error": "unknown action"}


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
        q = parse_qs(self.path.partition("?")[2])
        if "action" in q:
            if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
                return self.reply(403, {"error": "wrong site"})
            try:
                bn = int(self.headers.get("Content-Length", "0") or "0")
                if 0 < bn <= 4096:
                    self.rfile.read(bn)
            except ValueError:
                pass
            try:
                code, obj = tmux_act(q)
            except (subprocess.SubprocessError, OSError):
                code, obj = 503, {"error": "tmux refused"}
            return self.reply(code, obj)
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
        last[0] = time.time()
        path, _, query = self.path.partition("?")
        want = parse_qs(query).get("history")
        if path != "/term/upload" or not want:
            return self.reply(405, {"error": "post a picture"})
        if not PASSWORD or not hmac.compare_digest(self.headers.get("Authorization", ""), EXPECT):
            return self.reply(401, {"error": "sign in to the terminal first"})
        # Only the terminal page itself may read this. Its sign-in cookie is
        # same-site only and no cross-origin header is ever sent, so another
        # site could not read the reply; a fetch that says it comes from
        # elsewhere is refused outright as well.
        if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return self.reply(403, {"error": "wrong site"})
        try:
            text = history(int(want[0]))
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError):
            return self.reply(503, {"error": "the terminal session could not be read"})
        self.reply(200, {"text": text})


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
