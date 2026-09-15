from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).parent
CERTS = ROOT / ".certs"
CERT = CERTS / "cert.pem"
KEY = CERTS / "key.pem"


def local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main():
    CERTS.mkdir(exist_ok=True)
    address = local_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, address)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address(address)), x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    KEY.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    CERT.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    print(f"Generated local certificate for {address}")


if __name__ == "__main__":
    main()
