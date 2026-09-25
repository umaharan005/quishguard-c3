"""QuishGuard scan app: a phone scan page and an alert portal, served by FastAPI.

    python -m quishguard.app --models-dir models

  /          phone page: take a photo of a QR code -> tier, score, reasons, heatmap
  /portal    analyst portal: every scan, live; Phishing/Critical raise an alert
  /api/...   JSON API (POST /api/scan takes the raw image bytes)

No "from __future__ import annotations" here: FastAPI must see the real Request
type (imported inside create_app) to pass the request in.

The decoded link is never opened, by the server or by the pages: they show it
defanged (hxxps://example[.]com) and never as a clickable link.
"""
import argparse
import secrets
import socket
from pathlib import Path

from quishguard import config
from quishguard.app.devices import DeviceRegistry
from quishguard.app.service import MAX_BYTES, BadImage, ScanService
from quishguard.app.store import ScanStore

STATIC = Path(__file__).parent / "static"
ASSETS = {"ui.css": "text/css", "ui.js": "text/javascript"}      # shared design system


def create_app(service: ScanService, key: str | None = None, connect_url: str | None = None,
               devices: DeviceRegistry | None = None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="QuishGuard C3", docs_url=None, redoc_url=None)
    devices = devices or DeviceRegistry()

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
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
                                                   "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                                                   "font-src 'self' https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'")
        resp.headers["Permissions-Policy"] = "camera=(self)"
        return resp

    @app.get("/", response_class=HTMLResponse)
    def scan_page():
        return (STATIC / "scan.html").read_text(encoding="utf-8")

    @app.get("/portal", response_class=HTMLResponse)
    def portal_page():
        return (STATIC / "portal.html").read_text(encoding="utf-8")

    @app.get("/static/{name}")
    def asset(name: str):
        if name not in ASSETS:
            raise HTTPException(404)
        return FileResponse(STATIC / name, media_type=ASSETS[name])

    @app.post("/api/devices/ping")
    async def device_ping(request: Request):
        check(request)
        body = await request.json()
        did = str(body.get("id", ""))
        if not devices.valid(did):
            raise HTTPException(400, "Bad device id.")
        return devices.ping(did, request.headers.get("user-agent", ""), bool(body.get("reconnect")))

    @app.get("/api/devices")
    def device_list(request: Request):
        check(request)
        return devices.list()

    @app.post("/api/devices/{did}/disconnect")
    def device_disconnect(did: str, request: Request):
        check(request)
        if not devices.disconnect(did):
            raise HTTPException(404, "Unknown device.")
        return devices.list()

    @app.get("/api/health")
    def health():
        return {"ok": True, "key_required": key is not None, "connect_url": connect_url}

    @app.get("/connect.png")
    def connect_png(request: Request):
        """QR code of the phone scan page, shown on the portal (demo: point the phone at the laptop)."""
        import cv2
        from fastapi.responses import Response
        check(request)
        if not connect_url:
            raise HTTPException(404)
        img = cv2.QRCodeEncoder.create().encode(connect_url)
        img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", img)
        return Response(buf.tobytes(), media_type="image/png")

    @app.post("/api/scan")
    async def scan(request: Request):
        check(request)
        if int(request.headers.get("content-length") or 0) > MAX_BYTES:
            raise HTTPException(413, "Image is larger than 15 MB.")
        did = request.headers.get("x-device-id", "")
        ua = request.headers.get("user-agent", "")
        if devices.valid(did):
            if not devices.allowed(did):
                raise HTTPException(403, "This phone was disconnected from the alert portal. Tap Reconnect to scan again.")
            device = devices.count_scan(did, ua)
        else:
            device = request.headers.get("x-device", "") or "phone"
        data = await request.body()
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

    @app.post("/api/scans/{seq}/status")
    async def set_status(seq: int, request: Request):
        check(request)
        body = await request.json()
        if not service.store.set_status(seq, str(body.get("status", "")), str(body.get("note", ""))):
            raise HTTPException(400, "That status is not allowed for this scan.")
        return service.store.get(seq)

    @app.get("/api/export.csv")
    def export(request: Request):
        """All scans as CSV (for the report / error analysis). Links are defanged."""
        import csv
        import io
        from fastapi.responses import Response
        check(request)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["seq", "time_utc", "device", "tier", "score", "url_score", "url_score_model", "visual_score",
                    "visual_mode", "decoded", "payload_type", "link_defanged", "tranco_rank", "status", "note", "latency_ms"])
        for r in service.store.all_rows():
            a, d, st = r["result"], r["result"].get("decoded") or {}, r["result"].get("streams") or {}
            w.writerow([r["seq"], r["at_utc"], r["device"], r["tier"], r["score"], st.get("url"), a.get("url_score_model"),
                        st.get("visual"), a.get("visual_mode"), d.get("ok"), d.get("payload_type"),
                        d.get("text_defanged"), a.get("tranco_rank"), r["status"], r["note"], a.get("latency_ms")])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=quishguard_scans.csv"})

    @app.get("/files/{name}")
    def files(name: str, request: Request):
        check(request)
        p = service.file_path(name)
        if p is None:
            raise HTTPException(404)
        return FileResponse(p)

    return app


def self_signed_cert(data_dir: Path, ips: list[str]) -> dict:
    """Make (once) a self-signed certificate for this laptop's addresses. Phones need HTTPS
    before a web page may use the live camera; the photo button works without it."""
    import datetime
    import ipaddress
    cert_p, key_p = data_dir / "cert.pem", data_dir / "key.pem"
    if not (cert_p.exists() and key_p.exists()):
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.x509.oid import NameOID
        except ImportError:
            raise SystemExit("--https needs the 'cryptography' package: pip install cryptography")
        data_dir.mkdir(parents=True, exist_ok=True)
        k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "QuishGuard local")])
        alt = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips + ["127.0.0.1"]]
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=365))
                .add_extension(x509.SubjectAlternativeName(alt), critical=False).sign(k, hashes.SHA256()))
        key_p.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                          serialization.NoEncryption()))
        cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return {"ssl_certfile": str(cert_p), "ssl_keyfile": str(key_p)}


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
    ap.add_argument("--photo-clean", action="store_true",
                    help="binarise the photo crop before the visual model (default: raw crop)")
    ap.add_argument("--photo-visual", choices=["advisory", "full"], default="advisory",
                    help="camera photos: image check shown but not driving the tier (advisory, default) or fused (full)")
    ap.add_argument("--tranco", type=Path, default=None,
                    help="Tranco list (rank,domain) for the popular-platform check (default: models/tranco_top1m.csv if present)")
    ap.add_argument("--visual", default=None,
                    help="visual model: patchcore_adapted (default when present) or patchcore (original)")
    ap.add_argument("--key", default=None, help="optional access key; open the pages with ?key=YOURKEY")
    ap.add_argument("--webhook", default=None, help="optional URL that receives every Phishing/Critical alert (group aggregator)")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--https", action="store_true",
                    help="serve over HTTPS with a self-signed certificate: enables the live camera view on phones "
                         "(the phone shows a one-time certificate warning)")
    args = ap.parse_args(argv)

    import uvicorn
    from quishguard.pipeline import QuishGuard

    url_model = args.url_model or ("url_model_v2_portable.joblib"
                                   if (args.models_dir / "url_model_v2_portable.joblib").exists() else "url_model_v2.joblib")
    visual = args.visual or ("patchcore_adapted" if (args.models_dir / "visual_patchcore_adapted.pt").exists() else "patchcore")
    print("Loading models from", args.models_dir, "(URL model:", url_model + ", visual model:", visual + ") ...")
    qg = QuishGuard(models_dir=args.models_dir, url_model=url_model, w_url=args.w_url,
                    tamper_floor=0.0 if args.no_floor else 0.6, device=args.device,
                    photo_clean=args.photo_clean, visual_arch=visual,
                    photo_visual=args.photo_visual,
                    tranco_csv=args.tranco or (args.models_dir / "tranco_top1m.csv"))
    service = ScanService(qg, ScanStore(args.data_dir / "scans.db"), args.data_dir / "scans", args.webhook)
    q = f"?key={args.key}" if args.key else ""
    ips = lan_addresses()
    scheme = "https" if args.https else "http"
    app = create_app(service, args.key, connect_url=f"{scheme}://{ips[0]}:{args.port}/{q}" if ips else None)
    ssl = self_signed_cert(args.data_dir, ips) if args.https else {}

    print("Popular-platform check:", "on" if qg.reputation else "off (put tranco_top1m.csv in the models folder)")
    print("\nQuishGuard is running. Keep this window open.")
    for ip in ips or ["<this computer's IP>"]:
        print(f"  Phone scan page : {scheme}://{ip}:{args.port}/{q}")
        print(f"  Alert portal    : {scheme}://{ip}:{args.port}/portal{q}")
    print(f"  On this laptop  : {scheme}://localhost:{args.port}/portal{q}\n")
    if args.https:
        print("  HTTPS with a self-signed certificate: on the phone choose 'Advanced' -> 'Proceed' once.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
