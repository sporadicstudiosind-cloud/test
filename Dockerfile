# =============================================================================
# Personal TLS Proxy — Dockerfile
# Build:  docker build -t tls-proxy .
# Run:    docker compose up   (see docker-compose.yml)
# =============================================================================

FROM node:22-alpine AS base

# Install security updates
RUN apk update && apk upgrade --no-cache && apk add --no-cache dumb-init

WORKDIR /app

# Install dependencies first (leverages layer cache)
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy application source
COPY server.js ./

# ---------------------------------------------------------------------------
# Security hardening:
# - Run as non-root user
# - Read-only filesystem-friendly (certs and .env are volume-mounted)
# ---------------------------------------------------------------------------
RUN addgroup -S proxygroup && adduser -S proxyuser -G proxygroup
RUN chown -R proxyuser:proxygroup /app
USER proxyuser

EXPOSE 8443

# dumb-init reaps zombie processes and forwards signals correctly
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server.js"]

# Health check — verifies the port is open (not full TLS handshake)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node -e "require('net').createConnection(process.env.PORT||8443,'127.0.0.1').on('connect',()=>process.exit(0)).on('error',()=>process.exit(1))"
