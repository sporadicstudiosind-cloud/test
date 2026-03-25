# Personal TLS Proxy

A self-hosted HTTPS forward proxy that encrypts all traffic between your device
and the proxy server. Designed for use on untrusted WiFi networks (coffee shops,
hotels, airports). Runs on any cheap VPS or home server.

```
[Your Device] ──TLS encrypted──► [Your VPS :8443] ──► [Internet]
```

## What it does

- Encrypts the client → proxy channel (TLS 1.2+), so untrusted networks cannot
  read your traffic
- Supports HTTP forward proxying and HTTPS CONNECT tunneling
- WebSocket over TLS (WSS) works transparently through the CONNECT tunnel
- Basic proxy authentication (username + password)
- Single Node.js file, one runtime dependency (`dotenv`)

---

## Quick Start

### Prerequisites

- Node.js ≥ 18
- `openssl` command (available on macOS, Linux; Windows users: use Git Bash or WSL)

### 1. Install dependencies

```bash
npm install
```

### 2. Generate a TLS certificate

```bash
npm run gen-cert
```

This creates `certs/cert.pem` and `certs/key.pem`. **Save the SHA-256
fingerprint printed at the end** — you'll need it when configuring browsers.

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and set a strong username and password:

```ini
PORT=8443
PROXY_USER=myuser
PROXY_PASS=your_very_strong_random_password
CERT_PATH=./certs/cert.pem
KEY_PATH=./certs/key.pem
```

Generate a strong password: `openssl rand -base64 24`

### 4. Start the proxy

```bash
npm start
```

The server prints its SHA-256 certificate fingerprint on startup — verify it
matches `certs/cert.pem` before trusting it in your browser.

---

## Docker Deployment

```bash
# 1. Generate certs and configure .env (steps 2–3 above)

# 2. Build and start
docker compose up -d

# 3. View logs
docker compose logs -f proxy

# 4. Stop
docker compose down
```

---

## Client Configuration

In all cases below, replace `YOUR_SERVER_IP` with your VPS IP address or domain.

### Firefox (recommended — easiest self-signed cert setup)

1. Open `about:preferences` → search for "proxy" → **Settings…**
2. Select **Manual proxy configuration**
3. HTTPS Proxy: `YOUR_SERVER_IP`   Port: `8443`
4. Check **"Also use this proxy for HTTPS"**
5. Click OK, then visit any HTTP site first — Firefox will prompt for proxy credentials
6. To trust the self-signed cert: navigate to `https://YOUR_SERVER_IP:8443` in
   Firefox, click **Advanced → Accept the Risk and Continue**, then close the tab.
   The proxy will now work for all HTTPS sites without further prompts.

### Chrome / Edge (macOS or Windows)

Chrome uses the system proxy — configure it at the OS level (see below).
Alternatively, use the **SwitchyOmega** extension:

1. Install [SwitchyOmega](https://chromewebstore.google.com/detail/proxy-switchyomega/padekgcemlokbadohgkifijomclgjgif)
2. Create a new profile → Protocol: **HTTPS**, Server: `YOUR_SERVER_IP`, Port: `8443`
3. Enable the profile
4. For the self-signed cert: visit `https://YOUR_SERVER_IP:8443` in Chrome,
   click **Advanced → Proceed** to add the cert exception.

### macOS — System-wide proxy

1. **System Settings → Network → [your interface] → Details → Proxies**
2. Enable **Secure Web Proxy (HTTPS)**
3. Server: `YOUR_SERVER_IP`   Port: `8443`
4. Username/Password: your proxy credentials
5. Trust the certificate:
   ```bash
   # Add to macOS Keychain (run once)
   sudo security add-trusted-cert -d -r trustRoot \
     -k /Library/Keychains/System.keychain certs/cert.pem
   ```
   Or open Keychain Access, drag `cert.pem` in, double-click it, expand
   **Trust**, set **"When using this certificate"** to **Always Trust**.

### Windows — System-wide proxy

1. **Settings → Network & Internet → Proxy → Manual proxy setup**
2. Turn on **Use a proxy server**
3. Address: `YOUR_SERVER_IP`   Port: `8443`
4. Trust the certificate: double-click `cert.pem` → Install Certificate →
   Local Machine → Place in **Trusted Root Certification Authorities**

### iOS — Per-WiFi proxy

1. **Settings → Wi-Fi → [network] → Configure Proxy → Manual**
2. Server: `YOUR_SERVER_IP`   Port: `8443`
3. Authentication: enable, enter username/password
4. Trust the cert: download `cert.pem` to the device, open Settings → Profile
   Downloaded → Install, then **Settings → General → About → Certificate Trust
   Settings** → enable full trust for the cert.

### PAC file (advanced — proxy only selected traffic)

Create a `proxy.pac` file and serve it from a web server:

```javascript
function FindProxyForURL(url, host) {
  // Route everything through the proxy
  return "HTTPS YOUR_SERVER_IP:8443";
}
```

Configure browsers/OS to use **Automatic proxy configuration** with the URL of
your PAC file.

### Testing from the command line

```bash
# HTTP request
curl -x https://myuser:mypass@YOUR_SERVER_IP:8443 --proxy-insecure \
  http://httpbin.org/ip

# HTTPS request (CONNECT tunnel)
curl -x https://myuser:mypass@YOUR_SERVER_IP:8443 --proxy-insecure \
  https://httpbin.org/ip

# Test 407 rejection
curl -x https://wronguser:wrongpass@YOUR_SERVER_IP:8443 --proxy-insecure \
  http://example.com
# → should return: 407 Proxy Authentication Required
```

`--proxy-insecure` tells curl to skip verification of the proxy's self-signed
cert. Browsers handle this through the certificate trust steps above.

---

## VPS Deployment (DigitalOcean / Linode / Vultr)

### 1. Create a server

- Provider: DigitalOcean, Linode, Vultr, or any VPS
- Plan: smallest available ($4–6/month, 1 vCPU, 512 MB RAM is plenty)
- OS: Ubuntu 22.04 LTS
- Region: closest to you (or where you want your traffic to appear from)

### 2. Initial server setup

```bash
# SSH in
ssh root@YOUR_SERVER_IP

# Create a non-root user
adduser proxy
usermod -aG sudo proxy
su - proxy

# Update system
sudo apt update && sudo apt upgrade -y
```

### 3. Install Node.js 22

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # should print v22.x.x
```

### 4. Deploy the proxy

```bash
# Clone the repo (or scp the files)
git clone https://github.com/youruser/tls-proxy.git
cd tls-proxy

npm install
npm run gen-cert
cp .env.example .env
nano .env   # set PROXY_USER and PROXY_PASS
```

### 5. Open firewall port

```bash
sudo ufw allow 8443/tcp
sudo ufw allow OpenSSH
sudo ufw enable
sudo ufw status
```

### 6. Run as a systemd service (auto-start on reboot)

```bash
sudo nano /etc/systemd/system/tls-proxy.service
```

Paste this (adjust `User` and `WorkingDirectory`):

```ini
[Unit]
Description=Personal TLS Proxy
After=network.target

[Service]
Type=simple
User=proxy
WorkingDirectory=/home/proxy/tls-proxy
ExecStart=/usr/bin/node server.js
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tls-proxy
# Harden the service
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/proxy/tls-proxy
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tls-proxy
sudo systemctl start tls-proxy
sudo systemctl status tls-proxy

# View logs
sudo journalctl -u tls-proxy -f
```

### 7. (Optional) Use port 443

To run on port 443 (standard HTTPS, easier to use on restrictive networks):

```bash
# Allow Node.js to bind to port 443 without root
sudo setcap 'cap_net_bind_service=+ep' $(which node)

# Change PORT=443 in .env
sudo ufw allow 443/tcp
sudo systemctl restart tls-proxy
```

### 8. (Optional) Use a real domain + Let's Encrypt cert

If you have a domain pointing to the VPS:

```bash
sudo apt install certbot
sudo certbot certonly --standalone -d proxy.yourdomain.com

# Update .env
CERT_PATH=/etc/letsencrypt/live/proxy.yourdomain.com/fullchain.pem
KEY_PATH=/etc/letsencrypt/live/proxy.yourdomain.com/privkey.pem
```

A real cert means no browser exceptions needed — configure HTTPS proxy with
your domain and it just works.

---

## Security Notes

### What this protects against
- **Passive WiFi eavesdropping** — all traffic from your device to the proxy is
  TLS-encrypted. Coffee shop / hotel network operators cannot read your
  unencrypted HTTP traffic or know what HTTPS sites you're visiting.

### What this does NOT protect against
- **DNS leaks** — if your OS resolves DNS outside the proxy, DNS queries are
  visible on the local network. For full protection, also configure encrypted
  DNS (DoH/DoT) or use a VPN instead of a proxy.
- **Traffic analysis** — an observer can still see your device connecting to the
  VPS IP, just not the content. This is not anonymization.
- **Your VPS provider** — they can see what IP addresses your proxy connects to.
  This is a personal convenience tool, not an anonymity tool.
- **HTTP sites without HTTPS** — while the client→proxy hop is encrypted, if
  the proxy connects to a plain HTTP destination, that last hop is in the clear
  (same as any network connection to an HTTP site).

### Hardening checklist

- [ ] Change `PROXY_USER` and `PROXY_PASS` from defaults before deploying
- [ ] Use a strong, random password: `openssl rand -base64 24`
- [ ] Record the certificate SHA-256 fingerprint and verify it when connecting
      from new devices
- [ ] Keep the server OS patched (`apt upgrade`)
- [ ] Consider restricting access by source IP in your firewall if your client
      IP is stable
- [ ] Use `LOG_LEVEL=minimal` in production to avoid logging destination URLs
- [ ] Rotate the certificate annually (or use Let's Encrypt for auto-renewal)

### Credential storage

- Never commit `.env` to version control
- The `.gitignore` excludes `.env` and `certs/` by default
- On mobile devices, avoid saving proxy credentials in iCloud/Google backups
  if your threat model requires it

---

## Limitations

- **Single-process, single-threaded** — suitable for personal use; not designed
  for shared/multi-user high-throughput scenarios
- **No connection pooling** — each proxied request opens a new TCP connection
  to the upstream server
- **No caching** — all requests go to the origin every time
- **Self-signed cert** — requires a one-time trust step per device/browser;
  use Let's Encrypt + a domain for a seamless experience
- **WebSocket** — WSS (WebSocket over TLS) works via CONNECT. Plain WS over
  HTTP `Upgrade` works via the HTTP proxy path. Both are supported.

---

## File Structure

```
.
├── server.js           Main proxy server
├── package.json        Node.js project manifest
├── .env.example        Configuration template
├── .gitignore          Excludes .env, certs/, node_modules/
├── generate-cert.sh    Self-signed certificate generator
├── Dockerfile          Container image definition
├── docker-compose.yml  One-command container deployment
└── README.md           This file
```

---

## License

MIT — use freely, at your own risk.
