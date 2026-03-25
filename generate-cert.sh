#!/usr/bin/env bash
# =============================================================================
# generate-cert.sh — Generate a self-signed TLS certificate for the proxy.
#
# Usage:
#   bash generate-cert.sh [COMMON_NAME]
#
# Outputs:
#   certs/cert.pem   — Certificate (share this with clients for fingerprint)
#   certs/key.pem    — Private key  (keep secret, never share)
# =============================================================================
set -euo pipefail

CN="${1:-personal-proxy}"
DAYS=3650  # 10 years
OUTDIR="certs"

command -v openssl >/dev/null 2>&1 || { echo "ERROR: openssl not found. Install it first."; exit 1; }

mkdir -p "$OUTDIR"

echo "Generating RSA-4096 self-signed certificate (CN=$CN, valid ${DAYS} days)…"

openssl req \
  -x509 \
  -newkey rsa:4096 \
  -keyout "$OUTDIR/key.pem" \
  -out    "$OUTDIR/cert.pem" \
  -sha256 \
  -days   "$DAYS" \
  -nodes \
  -subj   "/CN=${CN}/O=personal/C=US" \
  -addext "subjectAltName=IP:127.0.0.1,DNS:localhost,DNS:${CN}"

chmod 600 "$OUTDIR/key.pem"
chmod 644 "$OUTDIR/cert.pem"

echo ""
echo "Certificate written to $OUTDIR/cert.pem"
echo "Private key written to $OUTDIR/key.pem  (keep this secret!)"
echo ""
echo "SHA-256 fingerprint (save this — use it to verify your cert in browsers):"
openssl x509 -in "$OUTDIR/cert.pem" -noout -fingerprint -sha256 | sed 's/Fingerprint=/  /'
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and set PROXY_USER / PROXY_PASS"
echo "  2. npm start"
echo "  3. In your browser, add a security exception for this certificate"
echo "     using the fingerprint above."
