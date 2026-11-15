#!/usr/bin/env python3
"""claude-account — see which Claude account this server uses, and change it.

Everything on this box that talks to Claude (the owner's chat, the cron team
that builds Jack) signs in with ONE login, kept by the Claude CLI in
`~/.claude/.credentials.json`. This tool keeps a named copy of each login the
owner has signed in with, so changing account is one tap instead of a browser
round trip every time.

    claude-account            a menu (plain text; works in any terminal)
    claude-account --app      the same, and tells the web terminal page what to
                              show, so the page can draw its own sheet
    claude-account status     which account is in use
    claude-account save       keep a copy of the login in use
    claude-account use N      change to saved account N (or its email)
    claude-account add [email]  sign in with another account, then keep it
    claude-account --selftest   exercise save/use in a throwaway home

What a change does, in order: (1) the login in use is copied back to its own
slot, so the slot always holds its newest keys; (2) the chosen slot is written
over the live login, atomically; (3) the result is checked by asking Claude
for the account's usage. Nothing is ever printed from a key file. The slots
live in `~/.claude-accounts/` (owner-only), outside the repo.

Signing in is always the CLI's own `claude auth login`. This tool only watches
its output for the sign-in link and hands that to the page, because a 300-
character link wrapped over six rows cannot be tapped on a phone.
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HOME = os.environ.get("CLAUDE_ACCOUNT_HOME") or os.path.expanduser("~")
CLI = os.environ.get("CLAUDE_ACCOUNT_CLI") or "claude"
LIVE_CRED = os.path.join(HOME, ".claude", ".credentials.json")
LIVE_CFG = os.path.join(HOME, ".claude.json")
STORE = os.path.join(HOME, ".claude-accounts")
USAGE = os.environ.get("CLAUDE_ACCOUNT_USAGE") or "/home/opc/jackthelearner/scripts/claude_usage.py"
PY = "/data/venvs/jackthelearner/bin/python"
APP = False
SIGNIN_HOSTS = ("claude.com", "claude.ai", "console.anthropic.com", "platform.claude.com")


# ── talking to the page ──────────────────────────────────────────────────────
def tell_page(msg: dict) -> None:
    """One private escape sequence the web terminal understands (OSC 7777),
    wrapped so tmux hands it through untouched. Other terminals ignore it."""
    if not APP:
        return
    b = base64.b64encode(json.dumps(msg).encode()).decode()
    osc = f"\x1b]7777;{b}\x07"
    if os.environ.get("TMUX"):
        osc = "\x1bPtmux;" + osc.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    sys.stdout.write(osc)
    sys.stdout.flush()


def say(text: str = "") -> None:
    print(text, flush=True)


# ── files ────────────────────────────────────────────────────────────────────
def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_private(path: str, data: bytes) -> None:
    """Atomic, owner-only: a half-written key file would sign everyone out."""
    d = os.path.dirname(path)
    os.makedirs(d, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def slug(email: str) -> str:
    return re.sub(r"[^a-z0-9._@+-]", "_", email.lower())[:120]


# ── who is signed in ─────────────────────────────────────────────────────────
def status() -> dict:
    """The CLI's own answer. {} when it cannot say."""
    try:
        out = subprocess.run([CLI, "auth", "status", "--json"], capture_output=True, text=True, timeout=30).stdout
        d = json.loads(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    if not d.get("loggedIn") or not d.get("email"):
        return {}
    return {"email": d["email"], "plan": (d.get("subscriptionType") or d.get("planTier") or "").title(),
            "org": d.get("orgName") or ""}


def usage_line() -> str:
    """The account's weekly use, which is also the proof that its keys work."""
    try:
        py = PY if os.path.exists(PY) else sys.executable
        out = subprocess.run([py, USAGE], capture_output=True, text=True, timeout=90).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    rows = [re.sub(r"\s*\[[#\s]*\]\s*", " ", ln).strip() for ln in out.splitlines() if ln.startswith("week:")]
    return " · ".join(rows)


def saved() -> list:
    rows = []
    if os.path.isdir(STORE):
        for name in sorted(os.listdir(STORE)):
            meta = read_json(os.path.join(STORE, name, "account.json"))
            if meta and os.path.exists(os.path.join(STORE, name, "credentials.json")):
                rows.append(meta)
    return rows


def save_current(cur: dict = None) -> dict:
    """Copy the live login into its slot. Called before every change, so a
    slot never goes stale while its account is the one in use."""
    cur = cur or status()
    if not cur or not os.path.exists(LIVE_CRED):
        return {}
    d = os.path.join(STORE, slug(cur["email"]))
    os.makedirs(STORE, mode=0o700, exist_ok=True)
    os.chmod(STORE, 0o700)
    with open(LIVE_CRED, "rb") as f:
        write_private(os.path.join(d, "credentials.json"), f.read())
    cfg = read_json(LIVE_CFG) or {}
    meta = {"email": cur["email"], "plan": cur.get("plan", ""), "org": cur.get("org", ""),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "oauthAccount": cfg.get("oauthAccount")}
    write_private(os.path.join(d, "account.json"), json.dumps(meta).encode())
    return meta


def use(target: dict) -> tuple:
    """Change the live login to a saved one. Returns (ok, message)."""
    cur = status()
    if cur and cur["email"] == target["email"]:
        return True, f"Already using {target['email']}."
    if cur:
        save_current(cur)
    src = os.path.join(STORE, slug(target["email"]), "credentials.json")
    try:
        with open(src, "rb") as f:
            keys = f.read()
        json.loads(keys)
    except (OSError, ValueError):
        return False, f"The saved sign-in for {target['email']} cannot be read. Sign in again."
    if os.path.exists(LIVE_CRED):
        with open(LIVE_CRED, "rb") as f:
            write_private(os.path.join(STORE, ".before-last-change.json"), f.read())
    write_private(LIVE_CRED, keys)
    # The CLI keeps the account's name beside its settings; move that too, or
    # it would go on showing the old name over the new keys.
    cfg = read_json(LIVE_CFG)
    if isinstance(cfg, dict) and target.get("oauthAccount"):
        cfg["oauthAccount"] = target["oauthAccount"]
        mode = os.stat(LIVE_CFG).st_mode & 0o777
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(LIVE_CFG))
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
        os.chmod(tmp, mode)
        os.replace(tmp, LIVE_CFG)
    now = status()
    if not now or now["email"] != target["email"]:
        return False, (f"Changed the keys, but Claude reports {now.get('email', 'nobody')} signed in. "
                       f"Sign in to {target['email']} again to repair it.")
    return True, f"Now using {target['email']}."


# ── signing in: the CLI's own flow, with the link handed to the page ─────────
def sign_in(email: str = "") -> bool:
    import fcntl
    import pty
    import select
    import struct
    import termios
    import tty

    cmd = [CLI, "auth", "login"] + (["--email", email] if email else [])
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)
    # A very wide screen for the child, so it never breaks the link across lines.
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 4000, 0, 0))
    seen, sent = b"", False
    stdin = sys.stdin.fileno()
    old = termios.tcgetattr(stdin) if os.isatty(stdin) else None
    try:
        if old:
            tty.setraw(stdin)
        while True:
            try:
                r, _, _ = select.select([fd, stdin], [], [], 0.5)
            except InterruptedError:
                continue
            if fd in r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
                if not sent:
                    seen = (seen + data)[-20000:]
                    m = re.search(rb"https://[A-Za-z0-9.-]+/[^\s\x1b\x07]+", re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]", b"", seen))
                    if m and re.search(rb"\s", seen[seen.find(m.group(0)) + len(m.group(0)):] or b""):
                        url = m.group(0).decode("ascii", "replace")
                        host = re.match(r"https://([^/]+)/", url).group(1).lower()
                        if host in SIGNIN_HOSTS or host.endswith(tuple("." + h for h in SIGNIN_HOSTS)):
                            tell_page({"t": "login", "url": url})
                            sent = True
            if stdin in r:
                try:
                    data = os.read(stdin, 4096)
                except OSError:
                    data = b""
                if not data:
                    break
                os.write(fd, data)
    finally:
        if old:
            termios.tcsetattr(stdin, termios.TCSADRAIN, old)
    try:
        _, st = os.waitpid(pid, 0)
        return os.WIFEXITED(st) and os.WEXITSTATUS(st) == 0
    except ChildProcessError:
        return False


def add(email: str = "") -> tuple:
    before = status()
    if before:
        save_current(before)
    say("\nSigning in with another Claude account.")
    say("Open the link, sign in with the account you want, then paste the code here.\n")
    ok = sign_in(email)
    after = status()
    if not ok or not after:
        return False, "Sign-in was not finished. Nothing was changed."
    save_current(after)
    if before and after["email"] == before["email"]:
        return True, f"That is the account already in use ({after['email']})."
    return True, f"Signed in. Now using {after['email']}."


# ── the menu ─────────────────────────────────────────────────────────────────
def menu() -> int:
    while True:
        cur = status()
        if cur:
            save_current(cur)
        rows = saved()
        say("\n  Claude account on this server\n")
        say(f"  In use:  {cur['email']}" + (f" · {cur['plan']}" if cur.get("plan") else "") if cur else "  Nobody is signed in.")
        say()
        for i, r in enumerate(rows, 1):
            mark = "   ← in use" if cur and r["email"] == cur["email"] else ""
            say(f"  {i}) {r['email']}" + (f" · {r['plan']}" if r.get("plan") else "") + mark)
        say("\n  a) Sign in with another account\n  q) Close\n")
        say("  Everything on this server follows the account you pick:\n  this chat, and the team that builds Jack.\n")
        tell_page({"t": "accounts", "cur": cur.get("email") if cur else None,
                   "list": [{"n": i, "email": r["email"], "plan": r.get("plan", ""),
                             "cur": bool(cur and r["email"] == cur["email"])} for i, r in enumerate(rows, 1)]})
        try:
            pick = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            say()
            tell_page({"t": "closed"})
            return 0
        if pick in ("q", "", "quit", "exit"):
            tell_page({"t": "closed"})
            return 0
        if pick == "a":
            ok, msg = add()
        elif pick.isdigit() and 1 <= int(pick) <= len(rows):
            tell_page({"t": "working", "email": rows[int(pick) - 1]["email"]})
            ok, msg = use(rows[int(pick) - 1])
        else:
            say("  Type a number, a, or q.")
            continue
        line = usage_line() if ok else ""
        if ok and not line:
            msg += " Its usage could not be read just now; if Claude asks you to sign in, choose 'a'."
        say(f"\n  {msg}" + (f"\n  This week: {line}" if line else ""))
        tell_page({"t": "done", "ok": ok, "msg": msg, "usage": line})
        if APP:
            try:
                input("\n  Press return to close. ")
            except (EOFError, KeyboardInterrupt):
                pass
            return 0 if ok else 1


# ── a check that needs no real account ───────────────────────────────────────
def selftest() -> int:
    tmp = tempfile.mkdtemp(prefix="claude-account-test-")
    try:
        os.makedirs(os.path.join(tmp, ".claude"))
        stub = os.path.join(tmp, "claude")
        with open(stub, "w") as f:
            f.write("#!/usr/bin/env python3\nimport json,os,sys\nh=os.environ['CLAUDE_ACCOUNT_HOME']\n"
                    "c=json.load(open(h+'/.claude.json'))\n"
                    "print(json.dumps({'loggedIn':True,'email':c['oauthAccount']['emailAddress'],'subscriptionType':'max'}))\n")
        os.chmod(stub, 0o755)
        env = dict(os.environ, CLAUDE_ACCOUNT_HOME=tmp, CLAUDE_ACCOUNT_CLI=stub, CLAUDE_ACCOUNT_USAGE="/nonexistent")

        def live(email, key):
            with open(os.path.join(tmp, ".claude", ".credentials.json"), "w") as f:
                json.dump({"claudeAiOauth": {"accessToken": key}}, f)
            with open(os.path.join(tmp, ".claude.json"), "w") as f:
                json.dump({"oauthAccount": {"emailAddress": email}, "other": 1}, f)

        def run(*a):
            return subprocess.run([sys.executable, os.path.abspath(__file__), *a], env=env, capture_output=True, text=True)

        def key():
            return json.load(open(os.path.join(tmp, ".claude", ".credentials.json")))["claudeAiOauth"]["accessToken"]

        checks = []
        live("a@example.com", "A1")
        run("save")
        live("b@example.com", "B1")
        run("save")
        r = run("use", "a@example.com")
        checks.append(("change to a saved account", r.returncode == 0 and key() == "A1"))
        cfg = json.load(open(os.path.join(tmp, ".claude.json")))
        checks.append(("the account's name moves with its keys, other settings kept",
                       cfg["oauthAccount"]["emailAddress"] == "a@example.com" and cfg["other"] == 1))
        # the account in use refreshes its keys; changing away must keep the NEW ones
        live("a@example.com", "A2")
        run("use", "2")
        checks.append(("change by number", key() == "B1"))
        run("use", "a@example.com")
        checks.append(("a slot keeps its newest keys", key() == "A2"))
        mode = os.stat(os.path.join(tmp, ".claude-accounts", "a@example.com", "credentials.json")).st_mode & 0o777
        checks.append(("saved keys are owner-only", mode == 0o600 and os.stat(os.path.join(tmp, ".claude-accounts")).st_mode & 0o777 == 0o700))
        r = run("use", "nobody@example.com")
        checks.append(("an unknown account is refused, nothing changed", r.returncode != 0 and key() == "A2"))
        out = run("status").stdout + r.stdout + r.stderr
        checks.append(("no key is ever printed", "A1" not in out and "A2" not in out and "B1" not in out))
        for name, ok in checks:
            print(("PASS  " if ok else "FAIL  ") + name)
        return 0 if all(ok for _, ok in checks) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list) -> int:
    global APP
    if "--app" in argv:
        APP = True
        argv = [a for a in argv if a != "--app"]
    if not argv:
        return menu()
    cmd = argv[0]
    if cmd == "--selftest":
        return selftest()
    if cmd == "--emit-login" and len(argv) > 1:      # tests: what the page does with a sign-in link
        APP = True
        tell_page({"t": "login", "url": argv[1]})
        return 0
    if cmd == "status":
        cur = status()
        say(f"{cur['email']}" + (f" · {cur['plan']}" if cur.get("plan") else "") if cur else "Nobody is signed in.")
        return 0 if cur else 1
    if cmd == "save":
        meta = save_current()
        say(f"Saved {meta['email']}." if meta else "Nobody is signed in; nothing to save.")
        return 0 if meta else 1
    if cmd == "use" and len(argv) > 1:
        rows = saved()
        pick = argv[1]
        target = rows[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(rows) else \
            next((r for r in rows if r["email"].lower() == pick.lower()), None)
        if not target:
            say(f"No saved account matches {pick!r}. Saved: " + (", ".join(r["email"] for r in rows) or "none"))
            return 1
        ok, msg = use(target)
        say(msg)
        return 0 if ok else 1
    if cmd == "add":
        ok, msg = add(argv[1] if len(argv) > 1 else "")
        say(msg)
        return 0 if ok else 1
    say(__doc__.split("\n\n")[2] if False else "usage: claude-account [--app] [status | save | use N|email | add [email]]")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
