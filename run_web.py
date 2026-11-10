"""run_web.py - One-command launcher for Mobile Face Recognition & Enrollment Web Server.

Detects local LAN IP, auto-generates self-signed SSL certificates (required by mobile
browsers for camera access), prints a scannable ASCII terminal QR code, and launches
the Uvicorn ASGI server.
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_web")

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent
CERTS_DIR = PROJECT_ROOT / "certs"


def get_local_ip() -> str:
    """Detect the host laptop's local LAN IP address.

    Connects a dummy UDP socket towards a public DNS (8.8.8.8) to let the OS kernel
    select the outgoing network interface without sending actual network traffic.
    Falls back to hostname resolution or 127.0.0.1.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()

    # If loopback is returned, scan host addresses for a non-loopback IPv4
    if ip.startswith("127."):
        try:
            hostname = socket.gethostname()
            for addr in socket.gethostbyname_ex(hostname)[2]:
                if not addr.startswith("127."):
                    ip = addr
                    break
        except Exception:
            pass

    return ip


def generate_self_signed_cert(
    cert_dir: Path = CERTS_DIR,
    host_ip: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Generate self-signed SSL certificate and private key using `cryptography`.

    Mobile browsers (Safari on iOS, Chrome on Android) strictly require HTTPS for
    `navigator.mediaDevices.getUserMedia()`. Self-signed certificates must include
    Subject Alternative Names (SAN) for both the IP address and localhost to be accepted.

    Returns:
        Tuple of (cert_path, key_path).
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    # Reuse existing certificates if present, non-empty, and containing current host_ip in SAN
    if cert_path.exists() and key_path.exists():
        if cert_path.stat().st_size > 0 and key_path.stat().st_size > 0:
            should_reuse = True
            if host_ip and host_ip != "127.0.0.1":
                try:
                    from cryptography import x509

                    cert_data = cert_path.read_bytes()
                    existing_cert = x509.load_pem_x509_certificate(cert_data)
                    san_ext = existing_cert.extensions.get_extension_for_oid(
                        x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                    )
                    san_ips = [str(n.value) for n in san_ext.value if isinstance(n, x509.IPAddress)]
                    if host_ip not in san_ips:
                        logger.info(
                            "LAN IP %s not in existing SSL cert SAN (%s). Regenerating cert...",
                            host_ip,
                            san_ips,
                        )
                        should_reuse = False
                except Exception as exc:
                    logger.debug("Could not verify existing cert SAN: %s", exc)
                    should_reuse = False

            if should_reuse:
                logger.info("Using existing SSL certificate at %s", cert_path)
                return cert_path, key_path

    logger.info("Generating self-signed SSL certificate in %s ...", cert_dir)
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        logger.error(
            "The 'cryptography' library is required to generate SSL certificates. "
            "Please run: pip install cryptography"
        )
        raise exc

    # 1. Generate 2048-bit RSA Private Key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # 2. Subject and Issuer
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FACE 2.0"),
        x509.NameAttribute(NameOID.COMMON_NAME, "FACE 2.0 Web Server"),
    ])

    # 3. Subject Alternative Names (SAN) - Essential for mobile browser trust
    san_list: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
    ]
    try:
        hostname = socket.gethostname()
        if hostname and hostname != "localhost":
            san_list.append(x509.DNSName(hostname))
    except Exception:
        pass

    san_list.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))

    if host_ip and host_ip != "127.0.0.1":
        try:
            san_list.append(x509.IPAddress(ipaddress.ip_address(host_ip)))
        except ValueError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )

    # 4. Serialize Key & Certificate to PEM
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    logger.info("SSL Certificate successfully created:")
    logger.info("  Cert: %s", cert_path)
    logger.info("  Key:  %s", key_path)
    return cert_path, key_path


def get_qr_ascii(url: str, indent: int = 4) -> str:
    """Generate an ASCII QR code string for display in the terminal.

    Uses `qrcode` if installed; returns an indented block string with invert=True
    optimized for standard dark terminal backgrounds.
    """
    prefix = " " * indent
    try:
        import io
        import qrcode

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=1,
        )
        qr.add_data(url)
        qr.make(fit=True)

        buf = io.StringIO()
        qr.print_ascii(out=buf, invert=True)
        raw_qr = buf.getvalue().rstrip()

        # Indent each line
        indented_lines = [f"{prefix}{line}" for line in raw_qr.split("\n")]
        return "\n".join(indented_lines)
    except Exception as exc:
        logger.debug("QR code generation exception: %s", exc)
        return (
            f"{prefix}[QR code preview unavailable: install `qrcode` library via `pip install qrcode`]\n"
            f"{prefix}Please enter the URL directly on your phone: {url}"
        )


def print_banner(
    proto: str,
    host: str,
    port: int,
    lan_ip: str,
    qr_ascii: str,
    threshold: float,
    db_path: str,
) -> None:
    """Print an attractive, clear startup banner with URLs and QR code."""
    local_url = f"{proto}://localhost:{port}"
    mobile_url = f"{proto}://{lan_ip}:{port}"
    ssl_active = proto == "https"

    divider = "=" * 72
    subdivider = "-" * 72

    print(divider, flush=True)
    print("           FACE 2.0 - MOBILE RECOGNITION & ENROLLMENT SERVER        ", flush=True)
    print(divider, flush=True)
    print(f"  * Protocol:       {proto.upper()} ({'SSL Enabled' if ssl_active else 'Insecure HTTP'})", flush=True)
    print(f"  * Host Binding:   {host}:{port}", flush=True)
    print(f"  * Database:       {db_path}", flush=True)
    print(f"  * Match Threshold:{threshold:.2f} (ArcFace Cosine Similarity)", flush=True)
    print(subdivider, flush=True)
    print(f"  [>] Local URL:     {local_url}", flush=True)
    print(f"  [>] Mobile LAN URL:{mobile_url}", flush=True)
    print(subdivider, flush=True)
    print("  SCAN WITH SMARTPHONE CAMERA TO LAUNCH INSTANTLY:\n", flush=True)
    print(qr_ascii, flush=True)
    print("\n" + subdivider, flush=True)
    print("  MOBILE SETUP & USAGE TIPS:", flush=True)
    print("  1. Ensure phone & laptop are connected to the SAME Wi-Fi network.", flush=True)
    print(f"  2. Open the Mobile LAN URL: {mobile_url}", flush=True)
    if ssl_active:
        print("  3. Self-Signed SSL Security Warning (Normal on first connection):", flush=True)
        print("     - iOS Safari:   Tap 'Show Details' -> 'visit this website'.", flush=True)
        print("     - Android Chrome: Tap 'Advanced' -> 'Proceed to site (unsafe)'.", flush=True)
        print("     (Required by mobile browsers to unlock camera hardware permissions).", flush=True)
    print("  4. Tap 'Allow' when the browser asks for Camera access.", flush=True)
    print("  5. Toggle front/back camera, live HUD overlay, and enroll directly from phone!", flush=True)
    print(divider + "\n", flush=True)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    http: bool = False,
    threshold: float = 0.55,
    db_path: str = "users.db",
    ctx_id: int = -1,
    reload: bool = False,
) -> None:
    """Configure environment, print banners, and launch Uvicorn server."""
    # 1. Resolve Local LAN IP
    lan_ip = get_local_ip()

    # 2. Configure SSL
    proto = "http" if http else "https"
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    if not http:
        cert_p, key_p = generate_self_signed_cert(CERTS_DIR, host_ip=lan_ip)
        ssl_certfile = str(cert_p)
        ssl_keyfile = str(key_p)

    target_url = f"{proto}://{lan_ip}:{port}"

    # 3. Generate QR Code
    qr_ascii = get_qr_ascii(target_url, indent=4)

    # 4. Print Banner
    print_banner(
        proto=proto,
        host=host,
        port=port,
        lan_ip=lan_ip,
        qr_ascii=qr_ascii,
        threshold=threshold,
        db_path=db_path,
    )

    # 5. Export environment variables for web.server module configuration
    os.environ["FACE_THRESHOLD"] = str(threshold)
    os.environ["FACE_DB_PATH"] = str(db_path)
    os.environ["FACE_CTX_ID"] = str(ctx_id)

    # 6. Start Uvicorn
    try:
        import uvicorn
    except ImportError:
        logger.error("Uvicorn is required to run the server. Run: pip install uvicorn")
        sys.exit(1)

    uvicorn_kwargs = {
        "app": "web.server:app",
        "host": host,
        "port": port,
        "log_level": "info",
        "reload": reload,
    }

    if ssl_certfile and ssl_keyfile:
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

    logger.info("Starting Uvicorn server on %s:%d ...", host, port)
    try:
        uvicorn.run(**uvicorn_kwargs)
    except KeyboardInterrupt:
        logger.info("Server gracefully stopped by user.")


def main() -> None:
    """CLI Entrypoint for run_web.py."""
    parser = argparse.ArgumentParser(
        description="FACE 2.0 - Mobile Face Recognition & Enrollment Web Server Launcher"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Network interface to bind (default: '0.0.0.0')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run in plain HTTP mode without SSL (Caution: mobile browsers require HTTPS for camera)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Cosine similarity threshold for face recognition (default: 0.55)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="users.db",
        help="SQLite database path (default: 'users.db')",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=-1,
        help="Inference device context: -1 for CPU, >= 0 for GPU (default: -1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reloading on code changes (development)",
    )

    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        http=args.http,
        threshold=args.threshold,
        db_path=args.db,
        ctx_id=args.ctx_id,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
