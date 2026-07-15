import json
import re
import socket
import urllib.request
import urllib.error
import uuid
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

REDIS_HOST, REDIS_PORT = '127.0.0.1', 6379
COG_URL = "http://127.0.0.1:5000"
REDIS_MESSAGE_TTL=86400 # 24h
# Max cached prediction JSON (bulk string); ~200 MiB
REDIS_MAX_BULK_BYTES = 200 * 1024 * 1024
REDIS_MAX_LINE_BYTES = 65536  # max RESP line (+/-/$/:), long -ERR messages
REDIS_SOCKET_TIMEOUT = 120  # large payloads to loopback Redis

# Safe Redis key / Replicate-style prediction id (client may override POST id)
PREDICTION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

AUTH_TOKEN = os.environ.get("BRIDGE_AUTH_TOKEN", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "default_secret")

def bridge_log_ext(message):
    """HTTP edge: clients outside the pod hitting the bridge API."""
    print(f"[bridge ext] {message}", flush=True)

def bridge_log_int(message):
    """In-pod legs: Redis, Cog upstream HTTP, Cog→bridge webhook."""
    print(f"[bridge int] {message}", flush=True)

def bridge_log_boot(message):
    print(f"[bridge] {message}", flush=True)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def encode_redis_command(cmd_list):
    out = bytearray()
    out.extend(f"*{len(cmd_list)}\r\n".encode("ascii"))
    for arg in cmd_list:
        b = str(arg).encode("utf-8")
        out.extend(f"${len(b)}\r\n".encode("ascii"))
        out.extend(b)
        out.extend(b"\r\n")
    return bytes(out)

def redis_fill(sock, pending):
    chunk = sock.recv(65536)
    if not chunk:
        raise ConnectionError("Redis connection closed while reading")
    pending.extend(chunk)

def redis_read_line(sock, pending):
    while True:
        i = pending.find(b"\r\n")
        if i >= 0:
            if i > REDIS_MAX_LINE_BYTES:
                raise ValueError(f"RESP line exceeds max length ({REDIS_MAX_LINE_BYTES} bytes)")
            line = bytes(pending[:i])
            del pending[: i + 2]
            return line
        if len(pending) > REDIS_MAX_LINE_BYTES:
            raise ValueError(f"RESP line exceeds max length ({REDIS_MAX_LINE_BYTES} bytes)")
        redis_fill(sock, pending)

def redis_read_n(sock, pending, n):
    out = bytearray()
    while len(out) < n:
        need = n - len(out)
        if pending:
            take = min(need, len(pending))
            out.extend(pending[:take])
            del pending[:take]
        else:
            redis_fill(sock, pending)
    return bytes(out)

def redis_expect_crlf(sock, pending):
    while len(pending) < 2:
        redis_fill(sock, pending)
    if pending[:2] != b"\r\n":
        raise ValueError(f"expected CRLF after bulk string, got {pending[:8]!r}")
    del pending[:2]

def redis_read_reply(sock, pending):
    line = redis_read_line(sock, pending)
    if not line:
        raise ValueError("empty Redis reply line")
    kind = line[:1]
    if kind == b"+":
        return line[1:].decode("utf-8", errors="replace")
    if kind == b"-":
        raise RuntimeError(line[1:].decode("utf-8", errors="replace"))
    if kind == b"$":
        try:
            blen = int(line[1:])
        except ValueError as e:
            raise ValueError(f"invalid bulk string length line: {line!r}") from e
        if blen == -1:
            return None
        if blen < 0:
            raise ValueError(f"invalid bulk string length: {blen}")
        if blen > REDIS_MAX_BULK_BYTES:
            raise ValueError(
                f"bulk string length {blen} exceeds max ({REDIS_MAX_BULK_BYTES} bytes)"
            )
        payload = redis_read_n(sock, pending, blen)
        redis_expect_crlf(sock, pending)
        return payload.decode("utf-8")
    if kind == b":":
        try:
            return int(line[1:])
        except ValueError as e:
            raise ValueError(f"invalid integer reply line: {line!r}") from e
    raise ValueError(f"unsupported Redis reply: {line[:80]!r}")

def _redis_command_summary(cmd_list, value_bytes=None):
    if not cmd_list:
        return "?"
    c = cmd_list[0]
    if c == "GET" and len(cmd_list) > 1:
        return f"GET key={cmd_list[1]!r}"
    if c == "SET" and len(cmd_list) >= 3:
        if value_bytes is not None:
            return f"SET key={cmd_list[1]!r} value_bytes={value_bytes}"
        return f"SET key={cmd_list[1]!r}"
    return str(c)

def redis_execute(cmd_list, value_bytes=None):
    """Run one Redis command. Returns {"ok": True, "data": ...} or {"ok": False, "error": str}."""
    summary = _redis_command_summary(cmd_list, value_bytes=value_bytes)
    bridge_log_int(f"redis command start {summary}")
    pending = bytearray()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(REDIS_SOCKET_TIMEOUT)
            s.connect((REDIS_HOST, REDIS_PORT))
            s.sendall(encode_redis_command(cmd_list))
            data = redis_read_reply(s, pending)
            bridge_log_int(f"redis command success {summary}")
            return {"ok": True, "data": data}
    except Exception as e:
        bridge_log_int(f"redis command failed {summary} error={e!r}")
        return {"ok": False, "error": str(e)}

class ReplicateCompatibleBridge(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return

    def _peer_is_in_pod(self):
        host = self.client_address[0]
        if host in ("127.0.0.1", "::1"):
            return True
        if host.startswith("::ffff:127.0.0.1"):
            return True
        return False

    def _log_edge(self):
        return bridge_log_int if self._peer_is_in_pod() else bridge_log_ext

    def _parse_content_length(self):
        raw = self.headers.get("Content-Length")
        if raw is None:
            return 0
        raw = raw.strip()
        if not raw:
            return 0
        try:
            n = int(raw)
        except ValueError:
            return None
        if n < 0:
            return None
        return n

    def _send_json(self, status, payload, close_connection=False):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _drain_body(self, total):
        remaining = total
        drained = 0
        while remaining > 0:
            chunk = self.rfile.read(min(262144, remaining))
            if not chunk:
                break
            drained += len(chunk)
            remaining -= len(chunk)
        return drained

    def is_authorized(self):
        if self.path in ("/health", "/health-check"):
            return True

        if f"/{WEBHOOK_SECRET}/webhook" in self.path:
            return True

        auth_header = self.headers.get('Authorization')
        if auth_header == f"Bearer {AUTH_TOKEN}":
            return True
        return False

    def do_GET(self):
        if self.path == "/health":
            if not self.is_authorized():
                self.send_error(401, "Unauthorized"); return
            self.send_response(200); self.end_headers(); return
        if self.path == "/health-check":
            if not self.is_authorized():
                self.send_error(401, "Unauthorized"); return
            self.proxy_request("GET", silent=True); return

        client = self.client_address[0]
        edge = self._log_edge()
        edge(f"GET path={self.path!r} client={client}")
        if not self.is_authorized():
            edge(f"GET denied unauthorized path={self.path!r} client={client}")
            self.send_error(401, "Unauthorized"); return

        pred_id = self.path.strip('/').split('/')[-1]
        if not PREDICTION_ID_RE.match(pred_id):
            edge(
                f"GET invalid prediction_id={pred_id!r} client={client} "
                "reason=id_format"
            )
            self._send_json(
                400,
                {
                    "error": "invalid_prediction_id",
                    "detail": (
                        "prediction id must be 1-64 characters from "
                        "[a-zA-Z0-9_-] only"
                    ),
                },
                close_connection=True,
            )
            return
        edge(f"GET prediction lookup prediction_id={pred_id!r}")
        result = redis_execute(["GET", pred_id])
        if result["ok"]:
            if result["data"] is not None:
                rb = len(result["data"].encode("utf-8"))
                edge(f"GET cache hit prediction_id={pred_id!r} response_bytes={rb}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(result["data"].encode('utf-8'))
                return
            edge(f"GET cache miss prediction_id={pred_id!r} reason=redis_key_absent")
        else:
            edge(
                f"GET redis failure prediction_id={pred_id!r} error={result['error']!r}"
            )
        bridge_log_int(f"GET cog upstream proxy prediction_id={pred_id!r}")
        self.proxy_request("GET")

    def do_POST(self):
        client = self.client_address[0]
        edge = self._log_edge()
        edge(f"POST path={self.path!r} client={client}")
        if not self.is_authorized():
            edge(f"POST denied unauthorized path={self.path!r} client={client}")
            self.send_error(401, "Unauthorized"); return
        if f"/{WEBHOOK_SECRET}/webhook" in self.path:
            bridge_log_int("POST routed to cog webhook handler (in-pod)")
            self.handle_webhook()
        else:
            edge("POST routed to cog API proxy")
            self.proxy_request("POST")

    def handle_webhook(self):
        pred_id = parse_qs(urlparse(self.path).query).get("id", ["last"])[0]
        client = self.client_address[0]
        bridge_log_int(
            f"webhook POST received prediction_id={pred_id!r} client={client} path={self.path!r}"
        )
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            bridge_log_int(
                f"webhook rejected prediction_id={pred_id!r} reason=chunked_transfer_encoding_not_supported"
            )
            self._send_json(
                501,
                {
                    "error": "unsupported_transfer_encoding",
                    "detail": "Chunked Transfer-Encoding is not supported; send Content-Length",
                },
                close_connection=True,
            )
            drain_cap = 64 * 1024 * 1024
            chunk_sz = 65536
            drained = 0
            while drained < drain_cap:
                chunk = self.rfile.read(min(chunk_sz, drain_cap - drained))
                if not chunk:
                    break
                drained += len(chunk)
            bridge_log_int(
                f"webhook chunked 501 body drain prediction_id={pred_id!r} "
                f"drained_bytes={drained} hit_cap={drained >= drain_cap}"
            )
            return
        cl = self._parse_content_length()
        if cl is None:
            bridge_log_int(
                f"webhook rejected prediction_id={pred_id!r} reason=invalid_content_length "
                f"header={self.headers.get('Content-Length')!r}"
            )
            self._send_json(
                400,
                {
                    "error": "invalid_content_length",
                    "detail": "Content-Length must be a non-negative integer",
                },
                close_connection=True,
            )
            return
        bridge_log_int(f"webhook prediction_id={pred_id!r} content_length={cl}")
        if cl > REDIS_MAX_BULK_BYTES:
            bridge_log_int(
                f"webhook rejected prediction_id={pred_id!r} reason=payload_too_large "
                f"bytes={cl} max_bytes={REDIS_MAX_BULK_BYTES}"
            )
            detail = (
                f"payload size ({cl} bytes) exceeds max ({REDIS_MAX_BULK_BYTES} bytes)"
            )
            self._send_json(
                413,
                {"error": "payload_too_large", "detail": detail},
                close_connection=True,
            )
            drained = self._drain_body(cl)
            bridge_log_int(
                f"webhook body drain prediction_id={pred_id!r} expected_bytes={cl} "
                f"drained_bytes={drained} complete={drained >= cl}"
            )
            return
        if not PREDICTION_ID_RE.match(pred_id):
            bridge_log_int(
                f"webhook rejected prediction_id={pred_id!r} reason=invalid_id_format"
            )
            self._send_json(
                400,
                {
                    "error": "invalid_prediction_id",
                    "detail": (
                        "query id must be 1-64 characters from [a-zA-Z0-9_-] only"
                    ),
                },
                close_connection=True,
            )
            self._drain_body(cl)
            return
        body = self.rfile.read(cl).decode("utf-8")
        body_bytes = cl
        payload_status = None
        try:
            payload_status = json.loads(body).get("status")
        except Exception as ex:
            bridge_log_int(f"webhook payload JSON parse skipped prediction_id={pred_id!r} detail={ex!r}")
        bridge_log_int(
            f"webhook body read prediction_id={pred_id!r} body_bytes={body_bytes} "
            f"payload_status={payload_status!r}"
        )
        set_result = redis_execute(
            ["SET", pred_id, body, "EX", f"{REDIS_MESSAGE_TTL}"],
            value_bytes=cl,
        )
        if not set_result["ok"]:
            err = set_result["error"]
            bridge_log_int(f"webhook redis_set_failed prediction_id={pred_id!r} detail={err!r}")
            self._send_json(
                503,
                {"error": "redis_set_failed", "detail": err},
                close_connection=True,
            )
            return
        if set_result.get("data") != "OK":
            err = f"unexpected SET reply: {set_result.get('data')!r}"
            bridge_log_int(f"webhook redis_set_failed prediction_id={pred_id!r} detail={err!r}")
            self._send_json(
                503,
                {"error": "redis_set_failed", "detail": err},
                close_connection=True,
            )
            return
        bridge_log_int(
            f"webhook ok prediction_id={pred_id!r} redis_stored ttl_seconds={REDIS_MESSAGE_TTL}"
        )
        self.send_response(200)
        self.end_headers()

    def proxy_request(self, method, silent=False):
        cl = self._parse_content_length()
        if (
            cl is None
            and silent
            and self.path == "/health-check"
        ):
            cl = 0
        if cl is None:
            if not silent:
                bridge_log_int(
                    f"proxy rejected invalid Content-Length method={method} path={self.path!r} "
                    f"header={self.headers.get('Content-Length')!r}"
                )
            self._send_json(
                400,
                {
                    "error": "invalid_content_length",
                    "detail": "Content-Length must be a non-negative integer",
                },
                close_connection=True,
            )
            return
        req_body = self.rfile.read(cl) if cl else b""
        if not silent:
            bridge_log_int(
                f"cog proxy request method={method} path={self.path!r} body_bytes={len(req_body)}"
            )
        target_path = self.path
        target_method = method

        if method == "POST" and self.path.strip('/') == "predictions":
            try:
                data = json.loads(req_body)
                generated_id = data.get("id") or str(uuid.uuid4().hex)[:24]
                data["id"] = generated_id
                if "webhook" not in data :
                    data["webhook"] = f"http://localhost:8080/{WEBHOOK_SECRET}/webhook?id={generated_id}"
                    data["webhook_events_filter"] = ["start", "completed"]
                req_body = json.dumps(data).encode('utf-8')
                if not silent:
                    bridge_log_int(
                        f"cog proxy predictions POST injected webhook prediction_id={generated_id!r} "
                        f"outgoing_body_bytes={len(req_body)}"
                    )
            except Exception as ex:
                if not silent:
                    bridge_log_int(f"cog proxy predictions JSON transform skipped detail={ex!r}")

        url = f"{COG_URL}{target_path}"
        req = urllib.request.Request(url, data=req_body if target_method != "GET" else None, method=target_method)
        for k, v in self.headers.items():
            if k.lower() not in ['host', 'content-length', 'authorization']:
                req.add_header(k, v)

        if not silent:
            bridge_log_int(f"cog proxy forwarding method={method} url={url!r}")
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                resp_data = res.read()
                if not silent:
                    bridge_log_int(
                        f"cog proxy response method={method} path={self.path!r} cog_status={res.status} "
                        f"response_bytes={len(resp_data)}"
                    )
                self.send_response(res.status)
                for k, v in res.getheaders():
                    if k.lower() != 'content-length': self.send_header(k, v)
                self.send_header('Content-Length', str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
        except Exception as e:
            if not silent:
                bridge_log_int(f"cog proxy upstream error method={method} path={self.path!r} detail={e!r}")
            self.send_response(502); self.end_headers()

if __name__ == '__main__':
    bridge_log_boot(
        f"bridge starting listen=0.0.0.0:8080 cog_upstream={COG_URL!r} "
        f"redis={REDIS_HOST!r}:{REDIS_PORT} max_bulk_bytes={REDIS_MAX_BULK_BYTES}"
    )
    ThreadedHTTPServer(('0.0.0.0', 8080), ReplicateCompatibleBridge).serve_forever()
