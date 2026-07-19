# Deploying the Web UI over HTTPS

The docker-compose stack ships with a [Caddy](https://caddyserver.com) reverse
proxy that terminates TLS in front of the web UI and obtains free,
browser-trusted certificates automatically:

```
browser ──https 443──▶ caddy ──http──▶ onit-web:9000 (uvicorn)
                 80 ──▶ redirect to https
```

uvicorn itself never sees a certificate. Caddy forwards `X-Forwarded-Proto`
and the original `Host` header, and uvicorn runs with `proxy_headers=True`, so
OAuth redirect URIs and cookies come out as `https://<your-domain>/...`
automatically.

## Where the certificates come from

Free certificates are issued by **[Let's Encrypt](https://letsencrypt.org)**,
a nonprofit certificate authority run by the Internet Security Research Group
(ISRG). (The EFF does not issue certificates — they maintain
[Certbot](https://certbot.eff.org), a client for fetching Let's Encrypt
certificates. You do **not** need Certbot with this stack: Caddy has the ACME
client built in.) Caddy will also fall back to **ZeroSSL**, a second free CA,
if Let's Encrypt has an outage.

Certificates are valid for 90 days and Caddy renews them automatically about
30 days before expiry. They are stored in the `caddy-data` Docker volume —
keep that volume; if you recreate it repeatedly you can hit Let's Encrypt's
[rate limits](https://letsencrypt.org/docs/rate-limits/) (5 duplicate
certificates per week).

## Prerequisites

1. **A domain name** with a DNS `A` (and/or `AAAA`) record pointing at the
   server's public IP, e.g. `mychat.ai`. Let's Encrypt will not issue
   certificates for `localhost` or private IPs.
2. **Ports 80 and 443 open** inbound (cloud firewall / security group). Port
   80 is required for the ACME HTTP-01 challenge and the HTTP→HTTPS redirect.
3. **A folder on permanent storage** for the agent's working files, e.g. an
   SSD mount at `/data/sandbox`. The container runs as UID/GID 1000, so the
   folder must be writable by that user:

   ```bash
   sudo mkdir -p /data/sandbox
   sudo chown 1000:1000 /data/sandbox
   ```

## Configuration

Everything is driven by `.env` in the repo root (the same file that holds the
model/API settings):

```bash
# Public domain — this is the only variable required for HTTPS.
ONIT_DOMAIN=mychat.ai

# Agent working directory (data_path) on the host. Mounted into the
# containers at /home/onit/data. Default: /data/sandbox
ONIT_DATA_DIR=/data/sandbox

# Optional: document corpus for the local_search tool, mounted read-only
# into the containers at /home/onit/documents (the containers see it via
# ONIT_DOCUMENTS_PATH automatically). Absolute path — ~ is not expanded.
# Files must be readable by UID 1000. Default: /data/documents
ONIT_DOCUMENTS_DIR=/home/me/internal-data

# Optional. Only needed if the public URL cannot be derived from the request
# (e.g. an extra proxy in front of Caddy rewrites the Host header).
# ONIT_PUBLIC_URL=https://mychat.ai
```

Leave `ONIT_DOMAIN` unset for local testing: Caddy then serves `localhost`
with a self-signed certificate (your browser will warn — expected).

**Google OAuth**: in the Google Cloud Console, set the authorized redirect URI
to `https://<your-domain>/auth/callback` (no port). The app builds the same
URI from the forwarded headers, so login works unchanged.

Optional: to get certificate-expiry notices from Let's Encrypt, add a global
options block at the top of the `Caddyfile`:

```
{
    email you@example.com
}
```

## Install / start

```bash
docker compose up -d --build
docker compose logs -f caddy    # watch the certificate being obtained
```

A successful issuance logs `certificate obtained successfully` within a few
seconds. Common failures:

| Symptom in caddy logs | Cause |
|---|---|
| `DNS problem: NXDOMAIN` | The `A` record doesn't exist yet or hasn't propagated |
| `Timeout during connect` | Port 80 blocked by a firewall, or DNS points at the wrong IP |
| `too many certificates already issued` | Rate-limited — you recreated the `caddy-data` volume too often; wait, and stop deleting the volume |

Nothing else on the host may bind ports 80/443 (stop any existing
nginx/apache first).

## Smoke tests after install

Run these from any machine (replace `mychat.ai` with your domain):

```bash
# 1. All services up?  (on the server)
docker compose ps

# 2. HTTP redirects to HTTPS (expect 308 + Location: https://...)
curl -sI http://mychat.ai | head -3

# 3. HTTPS serves the app (expect HTTP/2 200 and text/html)
curl -sI https://mychat.ai | head -3

# 4. Certificate is valid, trusted, and fresh (issuer C=US, O=Let's Encrypt)
openssl s_client -connect mychat.ai:443 -servername mychat.ai </dev/null 2>/dev/null \
  | openssl x509 -noout -issuer -subject -dates

# 5. Auth endpoint responds over HTTPS
curl -s https://mychat.ai/auth/check
```

Then in a browser:

6. **Login flow**: open `https://mychat.ai` — the padlock must show a valid
   certificate — and sign in with Google. If Google shows a
   `redirect_uri_mismatch` error, the redirect URI in the Cloud Console
   doesn't match `https://mychat.ai/auth/callback` exactly.
7. **Streaming**: send a chat message. Tokens must appear incrementally as
   they are generated. If the reply arrives only as one final block, a proxy
   in the path is buffering the SSE stream (the shipped `Caddyfile` disables
   buffering with `flush_interval -1`).

And on the server, verify persistence:

```bash
# 8. Agent files land on the SSD and survive a restart
ls /data/sandbox                 # one subfolder per chat session
docker compose restart onit-web
ls /data/sandbox                 # still there

# 9. uvicorn is NOT reachable from outside (run from another machine; must
#    time out / refuse — only the loopback mapping on the server may connect)
curl -m 5 http://mychat.ai:9000 && echo "PROBLEM: port 9000 is public!"
```

Certificate renewal needs no test or cron job — Caddy renews automatically as
long as the container is running and ports 80/443 stay reachable.

## Alternatives to the built-in setup

- **certbot + nginx**: install EFF's Certbot (`certbot --nginx -d mychat.ai`)
  and proxy to `127.0.0.1:9000`. You must then disable proxy buffering
  (`proxy_buffering off;`) and raise `proxy_read_timeout` for the SSE routes,
  and keep the systemd renewal timer Certbot installs.
- **Port 80 blocked?** Use the ACME DNS-01 challenge instead: swap the Caddy
  image for one built with your DNS provider's plugin
  (e.g. `caddy-dns/cloudflare`) and add a `tls { dns ... }` block.
- **Bring your own certificate** (corporate CA, wildcard): mount the files
  into the caddy container and point the site at them:
  `tls /certs/fullchain.pem /certs/privkey.pem`.
