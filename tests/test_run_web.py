"""tests/test_run_web.py - Unit tests for run_web.py and main.py launcher options."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509

from run_web import (
    generate_self_signed_cert,
    get_local_ip,
    get_qr_ascii,
)


def test_get_local_ip() -> None:
    """Validate that local LAN IP detection returns a valid IPv4 address string."""
    ip = get_local_ip()
    assert isinstance(ip, str)
    assert len(ip) > 0
    # Must be parsable as an IPv4 address
    parsed = ipaddress.ip_address(ip)
    assert parsed.version == 4


def test_generate_self_signed_cert(tmp_path: Path) -> None:
    """Validate that SSL certificate generation creates valid PEM files with SAN."""
    cert_dir = tmp_path / "test_certs"
    test_ip = "192.168.1.100"

    cert_path, key_path = generate_self_signed_cert(cert_dir, host_ip=test_ip)

    # 1. Check file existence and non-zero size
    assert cert_path.exists()
    assert key_path.exists()
    assert cert_path.stat().st_size > 0
    assert key_path.stat().st_size > 0

    # 2. Parse certificate and inspect Subject Alternative Names (SAN)
    cert_data = cert_path.read_bytes()
    cert = x509.load_pem_x509_certificate(cert_data)

    san_ext = cert.extensions.get_extension_for_oid(x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
    assert san_ext is not None
    san_value = san_ext.value

    dns_names = [name.value for name in san_value if isinstance(name, x509.DNSName)]
    ip_addresses = [str(name.value) for name in san_value if isinstance(name, x509.IPAddress)]

    assert "localhost" in dns_names
    assert "127.0.0.1" in ip_addresses
    assert test_ip in ip_addresses

    # 3. Check certificate reuse: calling again should reuse existing files without regenerating
    cert_mtime = cert_path.stat().st_mtime_ns
    key_mtime = key_path.stat().st_mtime_ns

    cert2, key2 = generate_self_signed_cert(cert_dir, host_ip=test_ip)
    assert cert2 == cert_path
    assert key2 == key_path
    assert cert_path.stat().st_mtime_ns == cert_mtime
    assert key_path.stat().st_mtime_ns == key_mtime


def test_get_qr_ascii() -> None:
    """Validate that ASCII QR code string is generated with block elements."""
    url = "https://192.168.1.100:8000"
    qr_text = get_qr_ascii(url, indent=2)

    assert isinstance(qr_text, str)
    assert len(qr_text) > 100
    # Must contain lines with indentation
    lines = qr_text.splitlines()
    assert len(lines) > 5
    assert all(line.startswith("  ") or not line.strip() for line in lines)


def test_run_web_cli_defaults() -> None:
    """Validate CLI argument parsing defaults in run_web."""
    import run_web

    test_parser = argparse.ArgumentParser()
    test_parser.add_argument("--host", default="0.0.0.0")
    test_parser.add_argument("--port", type=int, default=8000)
    test_parser.add_argument("--http", action="store_true")
    test_parser.add_argument("--threshold", type=float, default=0.55)
    test_parser.add_argument("--db", default="users.db")
    test_parser.add_argument("--ctx-id", type=int, default=-1)

    args = test_parser.parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.http is False
    assert args.threshold == 0.55
    assert args.db == "users.db"
    assert args.ctx_id == -1


def test_main_cli_web_flag() -> None:
    """Validate that main.py parses --web and delegates to run_server."""
    from main import main

    with patch("sys.argv", ["main.py", "--web", "--port", "9000", "--threshold", "0.60"]):
        with patch("run_web.run_server") as mock_run_server:
            main()
            mock_run_server.assert_called_once_with(
                host="0.0.0.0",
                port=9000,
                http=False,
                threshold=0.60,
                db_path="users.db",
                ctx_id=-1,
                reload=False,
            )
