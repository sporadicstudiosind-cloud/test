'use strict';

require('dotenv').config();

const https = require('https');
const http  = require('http');
const net   = require('net');
const fs    = require('fs');
const crypto = require('crypto');
const { URL } = require('url');

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const PORT       = parseInt(process.env.PORT || '8443', 10);
const PROXY_USER = process.env.PROXY_USER || '';
const PROXY_PASS = process.env.PROXY_PASS || '';
const CERT_PATH  = process.env.CERT_PATH  || './certs/cert.pem';
const KEY_PATH   = process.env.KEY_PATH   || './certs/key.pem';
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS || '30000', 10);
const LOG_LEVEL  = process.env.LOG_LEVEL  || 'verbose'; // 'verbose' | 'minimal'

// Hop-by-hop headers that must not be forwarded (RFC 7230 §6.1)
const HOP_BY_HOP = new Set([
  'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailers', 'transfer-encoding', 'upgrade',
  'proxy-connection', // de-facto
]);

// ---------------------------------------------------------------------------
// Startup validation
// ---------------------------------------------------------------------------
function validateConfig() {
  const errors = [];
  if (!PROXY_USER) errors.push('PROXY_USER is required');
  if (!PROXY_PASS) errors.push('PROXY_PASS is required');
  if (!fs.existsSync(CERT_PATH)) errors.push(`CERT not found: ${CERT_PATH}`);
  if (!fs.existsSync(KEY_PATH))  errors.push(`KEY not found:  ${KEY_PATH}`);
  if (errors.length) {
    for (const e of errors) log('error', e);
    log('error', 'Run: npm run gen-cert  then copy .env.example to .env and fill in credentials.');
    process.exit(1);
  }
  if (PROXY_PASS.length < 12) {
    log('warn', 'PROXY_PASS is short — use at least 12 characters for security.');
  }
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
function log(level, ...args) {
  if (level === 'debug' && LOG_LEVEL !== 'verbose') return;
  const ts = new Date().toISOString();
  const tag = `[${ts}] [${level.toUpperCase().padEnd(5)}]`;
  // eslint-disable-next-line no-console
  console.log(tag, ...args);
}

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------
function checkAuth(req) {
  if (!PROXY_USER) return true; // auth disabled (not recommended)

  const header = req.headers['proxy-authorization'] || '';
  if (!header.startsWith('Basic ')) return false;

  let decoded;
  try {
    decoded = Buffer.from(header.slice(6), 'base64').toString('utf8');
  } catch {
    return false;
  }

  const colonIdx = decoded.indexOf(':');
  if (colonIdx === -1) return false;

  const user = decoded.slice(0, colonIdx);
  const pass = decoded.slice(colonIdx + 1);

  // Constant-time comparison to prevent timing attacks
  const userBuf     = Buffer.alloc(256); Buffer.from(user).copy(userBuf);
  const passBuf     = Buffer.alloc(256); Buffer.from(pass).copy(passBuf);
  const expUserBuf  = Buffer.alloc(256); Buffer.from(PROXY_USER).copy(expUserBuf);
  const expPassBuf  = Buffer.alloc(256); Buffer.from(PROXY_PASS).copy(expPassBuf);

  const userOk = crypto.timingSafeEqual(userBuf, expUserBuf);
  const passOk = crypto.timingSafeEqual(passBuf, expPassBuf);
  return userOk && passOk;
}

function send407(socket) {
  const body = 'Proxy authentication required';
  socket.write(
    `HTTP/1.1 407 Proxy Authentication Required\r\n` +
    `Proxy-Authenticate: Basic realm="proxy"\r\n` +
    `Content-Length: ${body.length}\r\n` +
    `Connection: close\r\n` +
    `\r\n` +
    body
  );
  socket.destroy();
}

function send407Response(res) {
  res.writeHead(407, {
    'Proxy-Authenticate': 'Basic realm="proxy"',
    'Content-Type': 'text/plain',
    'Connection': 'close',
  });
  res.end('Proxy authentication required');
}

// ---------------------------------------------------------------------------
// Header sanitization — strip hop-by-hop and proxy headers before forwarding
// ---------------------------------------------------------------------------
function sanitizeHeaders(rawHeaders) {
  const out = {};
  // Collect extra hop-by-hop headers listed in Connection: header
  const connHeader = (rawHeaders['connection'] || '').toLowerCase();
  const extra = new Set(connHeader.split(',').map(s => s.trim()).filter(Boolean));

  for (const [k, v] of Object.entries(rawHeaders)) {
    const lk = k.toLowerCase();
    if (HOP_BY_HOP.has(lk) || extra.has(lk)) continue;
    out[k] = v;
  }
  return out;
}

// ---------------------------------------------------------------------------
// HTTPS CONNECT handler — creates a raw TCP tunnel
// Used by: HTTPS, WebSocket over TLS (WSS), HTTP/2, etc.
// ---------------------------------------------------------------------------
function handleConnect(req, clientSocket, head) {
  if (!checkAuth(req)) {
    log('warn', `CONNECT 407 — ${req.url} (bad credentials)`);
    send407(clientSocket);
    return;
  }

  const [host, portStr] = req.url.split(':');
  const port = parseInt(portStr, 10) || 443;

  log('debug', `CONNECT ${host}:${port}`);

  const targetSocket = net.createConnection({ host, port }, () => {
    // Signal success to the client
    clientSocket.write('HTTP/1.1 200 Connection Established\r\n\r\n');

    // If the client sent data before we acknowledged, replay it
    if (head && head.length > 0) targetSocket.write(head);

    // Bidirectional pipe
    targetSocket.pipe(clientSocket);
    clientSocket.pipe(targetSocket);
  });

  targetSocket.setTimeout(TIMEOUT_MS, () => {
    log('debug', `CONNECT timeout — ${host}:${port}`);
    targetSocket.destroy();
    clientSocket.destroy();
  });

  targetSocket.on('error', (err) => {
    log('debug', `CONNECT target error — ${host}:${port} — ${err.message}`);
    if (!clientSocket.destroyed) {
      clientSocket.write(
        `HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n`
      );
      clientSocket.destroy();
    }
  });

  clientSocket.on('error', (err) => {
    log('debug', `CONNECT client error — ${host}:${port} — ${err.message}`);
    if (!targetSocket.destroyed) targetSocket.destroy();
  });

  clientSocket.on('end', () => {
    if (!targetSocket.destroyed) targetSocket.destroy();
  });

  targetSocket.on('end', () => {
    if (!clientSocket.destroyed) clientSocket.destroy();
  });
}

// ---------------------------------------------------------------------------
// HTTP forward proxy handler
// ---------------------------------------------------------------------------
function handleRequest(req, res) {
  // Only handle absolute-form URLs (proxy requests)
  if (!req.url.startsWith('http://') && !req.url.startsWith('https://')) {
    res.writeHead(400, { 'Content-Type': 'text/plain' });
    res.end('Bad Request: use absolute URL');
    return;
  }

  if (!checkAuth(req)) {
    log('warn', `HTTP 407 — ${req.method} ${req.url} (bad credentials)`);
    send407Response(res);
    return;
  }

  let parsed;
  try {
    parsed = new URL(req.url);
  } catch {
    res.writeHead(400, { 'Content-Type': 'text/plain' });
    res.end('Bad Request: invalid URL');
    return;
  }

  const isHttps  = parsed.protocol === 'https:';
  const agent    = isHttps ? https : http;
  const port     = parsed.port ? parseInt(parsed.port, 10)
                               : (isHttps ? 443 : 80);

  const options = {
    hostname: parsed.hostname,
    port,
    path:     parsed.pathname + parsed.search,
    method:   req.method,
    headers:  sanitizeHeaders(req.headers),
    timeout:  TIMEOUT_MS,
  };

  log('debug', `HTTP  ${req.method} ${parsed.hostname}:${port}${options.path}`);

  const proxyReq = agent.request(options, (proxyRes) => {
    log('debug', `HTTP  ${req.method} ${parsed.hostname} — ${proxyRes.statusCode}`);

    const responseHeaders = sanitizeHeaders(proxyRes.headers);
    res.writeHead(proxyRes.statusCode, responseHeaders);
    proxyRes.pipe(res, { end: true });

    proxyRes.on('error', (err) => {
      log('debug', `HTTP upstream response error — ${err.message}`);
      if (!res.headersSent) {
        res.writeHead(502);
      }
      res.end();
    });
  });

  proxyReq.on('timeout', () => {
    log('debug', `HTTP timeout — ${parsed.hostname}`);
    proxyReq.destroy(new Error('Upstream request timed out'));
  });

  proxyReq.on('error', (err) => {
    log('debug', `HTTP upstream error — ${parsed.hostname} — ${err.message}`);
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'text/plain' });
    }
    res.end(`502 Bad Gateway: ${err.message}`);
  });

  req.on('error', (err) => {
    log('debug', `HTTP client req error — ${err.message}`);
    if (!proxyReq.destroyed) proxyReq.destroy();
  });

  req.pipe(proxyReq, { end: true });
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------
function createServer() {
  const tlsOptions = {
    key:  fs.readFileSync(KEY_PATH),
    cert: fs.readFileSync(CERT_PATH),
    // Disable old/weak TLS versions
    minVersion: 'TLSv1.2',
    // Strong cipher suite preference
    ciphers: [
      'ECDHE-ECDSA-AES256-GCM-SHA384',
      'ECDHE-RSA-AES256-GCM-SHA384',
      'ECDHE-ECDSA-CHACHA20-POLY1305',
      'ECDHE-RSA-CHACHA20-POLY1305',
      'ECDHE-ECDSA-AES128-GCM-SHA256',
      'ECDHE-RSA-AES128-GCM-SHA256',
    ].join(':'),
    honorCipherOrder: true,
  };

  const server = https.createServer(tlsOptions, handleRequest);

  // CONNECT method for HTTPS tunneling
  server.on('connect', handleConnect);

  // Catch server-level TLS/socket errors (e.g. client sent garbage)
  server.on('clientError', (err, socket) => {
    log('debug', `Client error — ${err.message}`);
    if (!socket.destroyed) socket.destroy();
  });

  server.on('error', (err) => {
    log('error', `Server error — ${err.message}`);
    if (err.code === 'EADDRINUSE') {
      log('error', `Port ${PORT} is already in use.`);
      process.exit(1);
    }
  });

  return server;
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------
function setupShutdown(server) {
  const shutdown = (signal) => {
    log('info', `${signal} received — shutting down gracefully…`);
    server.close(() => {
      log('info', 'Server closed.');
      process.exit(0);
    });
    // Force-kill after 10 s if something hangs
    setTimeout(() => {
      log('warn', 'Forcing exit after timeout.');
      process.exit(1);
    }, 10_000).unref();
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT',  () => shutdown('SIGINT'));
  process.on('uncaughtException', (err) => {
    log('error', `Uncaught exception — ${err.message}`, err.stack);
    shutdown('uncaughtException');
  });
  process.on('unhandledRejection', (reason) => {
    log('error', `Unhandled rejection — ${reason}`);
  });
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
validateConfig();

const server = createServer();
setupShutdown(server);

server.listen(PORT, '0.0.0.0', () => {
  const cert = fs.readFileSync(CERT_PATH);
  const fingerprint = crypto
    .createHash('sha256')
    .update(cert)
    .digest('hex')
    .replace(/(.{2})(?=.)/g, '$1:')
    .toUpperCase();

  log('info', `TLS proxy listening on 0.0.0.0:${PORT}`);
  log('info', `Auth user : ${PROXY_USER}`);
  log('info', `TLS min   : TLSv1.2`);
  log('info', `Timeout   : ${TIMEOUT_MS}ms`);
  log('info', `Log level : ${LOG_LEVEL}`);
  log('info', `Cert SHA-256 fingerprint:`);
  log('info', `  ${fingerprint}`);
  log('info', `Ready. Configure your browser to use HTTPS proxy at YOUR_SERVER_IP:${PORT}`);
});
