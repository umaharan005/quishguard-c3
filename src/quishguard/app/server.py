"""QuishGuard scan app: a phone scan page and an alert portal, served by FastAPI.

    python -m quishguard.app --models-dir models

  /          phone page: take a photo of a QR code -> tier, score, reasons, heatmap
  /portal    analyst portal: every scan, live; Phishing/Critical raise an alert
  /api/...   JSON API (POST /api/scan takes the raw image bytes)

The decoded link is never opened, by the server or by the pages: they show it
defanged (hxxps://example[.]com) and never as a clickable link.
"""
from __future__ import annotations

import argparse
import secrets
import socket
from pathlib import Path

from quishguard import config
from quishguard.app.service import MAX_BYTES, BadImage, ScanService
from quishguard.app.store import ScanStore

STATIC = Path(__file__).parent / "static"


def create_app(service: ScanService, key: str | None = None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="QuishGuard C3", docs_url=None, redoc_url=None)

    def check(request: Request):
        if key is None:
            return
        given = request.headers.get("x-qg-key") or request.query_params.get("key") or ""
        if not secrets.compare_digest(given, key):
            raise HTTPException(401, "Wrong or missing access key.")

    @app.middleware("http")
    async def no_cache_and_safe_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' blob: data:; "
                                                   "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
        return resp

    @app.get("/", response_class=HTMLResponse)
    def scan_page():
        return (STATIC / "scan.html").read_text(encoding="utf-8")

    @app.get("/portal", response_class=HTMLResponse)
    def portal_page():
        return (STATIC / "portal.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        return {"ok": True, "key_required": key is not None}

    @app.post("/api/scan")
    async def scan(request: Request):
        check(request)
        if int(request.headers.get("content-length") or 0) > MAX_BYTES:
            raise HTTPException(413, "Image is larger than 15 MB.")
        data = await request.body()
        device = request.headers.get("x-device", "")
        try:
            return await run_in_threadpool(service.handle, data, device)
        except BadImage as e:
            raise HTTPException(400, str(e))

    @app.get("/api/scans")
    def scans(request: Request, since: int = 0, limit: int = 100):
        check(request)
        return service.store.since(since, min(max(limit, 1), 500))

    @app.get("/api/stats")
    def stats(request: Request):
        check(request)
        return service.store.stats()

    @app.post("/api/scans/{seq}/ack")
    def ack(seq: int, request: Request):
        check(request)
        if not service.store.acknowledge(seq):
            raise HTTPException(404, "No open alert with that number.")
        return service.store.get(seq)

    @app.get("/files/{name}")
    def files(name: str, request: Request):
        check(request)
        p = service.file_path(name)
        if p is None:
            raise HTTPException(404)
        return FileResponse(p)

    return app


def lan_addresses() -> list[str]:
    """This computer's address on the Wi-Fi (what the phone must type). Sends nothing."""
    out = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))        # UDP connect only picks a route; no packet is sent
        out.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and ip not in out:
                out.append(ip)
    except OSError:
        pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--url-model", default=None,
                    help="URL model file in --models-dir (default: url_model_v2_portable.joblib if present, else url_model_v2.joblib)")
    ap.add_argument("--data-dir", type=Path, default=config.REPO_ROOT / "data" / "app",
                    help="where scans.db and the scan photos/heatmaps are kept")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from the phone on the same Wi-Fi")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--w-url", type=float, default=0.60, help="URL weight (0.60 was chosen on val in Phase 6)")
    ap.add_argument("--no-floor", action="store_true", help="turn off the tamper floor")
    ap.add_argument("--key", default=None, help="optional access key; open the pages with ?key=YOURKEY")
    ap.add_argument("--webhook", default=None, help="optional URL that receives every Phishing/Critical alert (group aggregator)")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = ap.parse_args(argv)

    import uvicorn
    from quishguard.pipeline import QuishGuard

    url_model = args.url_model or ("url_model_v2_portable.joblib"
                                   if (args.models_dir / "url_model_v2_portable.joblib").exists() else "url_model_v2.joblib")
    print("Loading models from", args.models_dir, "(URL model:", url_model + ") ...")
    qg = QuishGuard(models_dir=args.models_dir, url_model=url_model, w_url=args.w_url,
                    tamper_floor=0.0 if args.no_floor else 0.6, device=args.device)
    service = ScanService(qg, ScanStore(args.data_dir / "scans.db"), args.data_dir / "scans", args.webhook)
    app = create_app(service, args.key)

    q = f"?key={args.key}" if args.key else ""
    print("\nQuishGuard is running. Keep this window open.")
    for ip in lan_addresses() or ["<this computer's IP>"]:
        print(f"  Phone scan page : http://{ip}:{args.port}/{q}")
        print(f"  Alert portal    : http://{ip}:{args.port}/portal{q}")
    print(f"  On this laptop  : http://localhost:{args.port}/portal{q}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
