# Web Server Deployment Guide

This guide covers deploying the OnIt web UI for production use, transitioning from the default development port (9000) to standard HTTP (80) and HTTPS (443) ports.

## Overview

By default, OnIt's web UI runs on port 9000 via Uvicorn. For production, you should:

1. Keep Uvicorn on an internal port (9000)
2. Use a reverse proxy (nginx or Caddy) to handle ports 80/443
3. Terminate TLS at the reverse proxy

```
Client (browser)
  │
  ▼
nginx / Caddy  ── port 80  (HTTP → redirect to HTTPS)
                ── port 443 (HTTPS, TLS termination)
  │
  ▼
Uvicorn (OnIt)  ── port 9000 (internal only)
```

---

## Option A: nginx + Let's Encrypt (Certbot)

### 1. Install nginx and Certbot

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx
```

**macOS (Homebrew):**
```bash
brew install nginx
brew install certbot
```

### 2. Configure nginx

Create `/etc/nginx/sites-available/onit`:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

# HTTPS reverse proxy
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # TLS hardening
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support: disable buffering so streamed tokens arrive live
        proxy_http_version 1.1;
        proxy_buffering off;

        # Timeout settings for long-running LLM responses
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # File upload size limit
    client_max_body_size 50M;
}
```

### 3. Enable the site and obtain a certificate

```bash
sudo ln -s /etc/nginx/sites-available/onit /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test configuration
sudo nginx -t

# Obtain TLS certificate
sudo certbot --nginx -d your-domain.com

# Reload nginx
sudo systemctl reload nginx
```

Certbot auto-renews certificates via a systemd timer. Verify with:
```bash
sudo certbot renew --dry-run
```

### 4. Start OnIt

```bash
onit --web --web-port 9000
```

OnIt stays on port 9000 internally. nginx handles 80/443.

---

## Option B: Caddy (automatic HTTPS)

Caddy automatically provisions and renews TLS certificates via Let's Encrypt.

### 1. Install Caddy

**Ubuntu/Debian:**
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

**macOS:**
```bash
brew install caddy
```

### 2. Configure Caddy

Create or edit `/etc/caddy/Caddyfile`:

```
your-domain.com {
    reverse_proxy 127.0.0.1:9000

    # File upload size limit
    request_body {
        max_size 50MB
    }
}
```

That's it. Caddy handles HTTP-to-HTTPS redirection and TLS certificates automatically.

### 3. Start Caddy and OnIt

```bash
sudo systemctl enable --now caddy
onit --web --web-port 9000
```

---

## Docker Deployment

### docker-compose.yml with nginx

```yaml
services:
  onit-web:
    build: .
    expose:
      - "9000"
    env_file: .env
    command: ["--web", "--web-port", "9000"]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - onit-web
    restart: unless-stopped
```

Create `nginx.conf` alongside docker-compose.yml:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://onit-web:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;
    }

    client_max_body_size 50M;
}
```

Place your TLS certificate files in a `certs/` directory:
```
certs/
├── fullchain.pem
└── privkey.pem
```

Then start:
```bash
docker compose up --build -d
```

### Docker with Caddy

```yaml
services:
  onit-web:
    build: .
    expose:
      - "9000"
    env_file: .env
    command: ["--web", "--web-port", "9000"]
    restart: unless-stopped

  caddy:
    image: caddy:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
    depends_on:
      - onit-web
    restart: unless-stopped

volumes:
  caddy_data:
```

Create `Caddyfile`:
```
your-domain.com {
    reverse_proxy onit-web:9000
}
```

---

## Google OAuth2 with HTTPS

When switching to HTTPS, update your Google Cloud OAuth configuration:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) > Credentials
2. Update **Authorized JavaScript origins**:
   - Remove: `http://YOUR_SERVER_IP:9000`
   - Add: `https://your-domain.com`
3. Update **Authorized redirect URIs**:
   - Add: `https://your-domain.com/auth/callback`

In your OnIt config file, no changes are needed for `web_port` since the reverse proxy handles the public-facing ports.

---

## Firewall Configuration

Lock down so only the reverse proxy reaches OnIt:

```bash
# Allow HTTP and HTTPS from anywhere
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Block direct access to Uvicorn from outside
sudo ufw deny 9000/tcp

# If running MCP servers, restrict to localhost only (they bind to 0.0.0.0 by default)
sudo ufw deny 18200:18204/tcp

sudo ufw enable
```

---

## Quick Reference

| Setup | HTTP (80) | HTTPS (443) | TLS Certificates | Effort |
|-------|-----------|-------------|-------------------|--------|
| nginx + Certbot | Manual redirect config | Manual config | Auto via Certbot | Medium |
| Caddy | Automatic | Automatic | Automatic | Low |
| Docker + nginx | Via compose | Via compose | Manual (mount certs) | Medium |
| Docker + Caddy | Via compose | Via compose | Automatic | Low |
