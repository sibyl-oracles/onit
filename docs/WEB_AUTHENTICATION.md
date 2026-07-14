# Web UI Google OAuth2 Authentication

The OnIt web UI is gated by Google login. Every chat session starts with a
Google OAuth2 sign-in, only Google-hosted mail accounts are accepted, and
each session is private to the account that created it.

Step-by-step credential setup lives in the README under
[`onit serve web`](../README.md#onit-serve-web). This document covers how the
flow works, the configuration reference, and troubleshooting.

## Why authentication is required

The web server binds to `0.0.0.0`, so it is reachable from any network
interface. Without login:

- Anyone who can reach the port can drive the agent
- Uploaded and generated files are accessible to anyone
- There is no session isolation between users

For this reason `onit serve web` **refuses to start** unless Google OAuth2
credentials are configured (or login is explicitly disabled — see
[Disabling authentication](#disabling-authentication)).

## How the login flow works

OnIt implements the standard OAuth2 authorization-code flow with PKCE,
entirely server-side:

1. The browser opens `/` — the SPA sees `authenticated: false` from
   `/api/config` and shows only a **Sign in with Google** button.
2. The button links to `/auth/login`, which generates a CSRF `state` token
   and a PKCE verifier/challenge, then redirects to
   `accounts.google.com` requesting the `openid email profile` scopes.
3. The user picks a Google account; Google redirects back to
   `/auth/callback` with a one-time authorization code.
4. OnIt verifies the `state`, exchanges the code (with the PKCE verifier and
   client secret) for an ID token, and cryptographically verifies that token
   against Google's public keys and the configured client ID.
5. The account is accepted only if:
   - the email is **verified** by Google, and
   - it is **Google-hosted mail**: a Gmail address
     (`@gmail.com` / `@googlemail.com`) or a Google Workspace account (the
     ID token's `hd` hosted-domain claim matches the email's domain), and
   - it matches `web_allowed_emails`, when that list is configured.
6. On success OnIt sets an `onit_auth` httponly cookie (24-hour lifetime) and
   redirects to the chat. A session cookie is only issued after this point —
   unauthenticated visitors never receive one.

`/auth/logout` revokes the server-side session and clears the cookie.

## Per-user session isolation

Chat sessions are owned by the authenticated email:

- The session list (`/api/sessions`) shows only your own sessions.
- Reading, renaming, deleting, or downloading files from another user's
  session returns 404, even with a valid login.
- A session ID that belongs to someone else silently resolves to a fresh
  session instead.
- Sessions created before authentication was enabled are unowned; the first
  authenticated user to open one claims it, and it is locked to them
  thereafter.

Ownership is recorded in the session index (`~/.onit/sessions/index.json`)
and survives server restarts.

## Configuration reference

| Key | Meaning | Default |
|-----|---------|---------|
| `web_google_client_id` | OAuth client ID (`*.apps.googleusercontent.com`) | — (keychain via `onit setup`, or `GOOGLE_CLIENT_ID` env var) |
| `web_google_client_secret` | OAuth client secret (`GOCSPX-…`) | — (keychain via `onit setup`, or `GOOGLE_CLIENT_SECRET` env var) |
| `web_allowed_emails` | Extra allowlist: exact addresses or `"*@domain"` patterns | unset (any Gmail/Workspace account) |
| `web_require_auth` | Set `false` to allow running without login | `true` |

Credentials are resolved in order: config YAML → environment variable →
OS keychain (stored by `onit setup`). Verify what is set with
`onit setup --show`.

Example config:

```yaml
web: true
web_port: 9000
web_allowed_emails:
  - alice@gmail.com
  - "*@sibyl.ai"
```

## Session management

- Login sessions last 24 hours, held in server memory (a restart logs
  everyone out; chat history is preserved on disk).
- OAuth flow state (PKCE verifier + CSRF token) expires after 10 minutes and
  is single-use.
- Chat history is keyed by the `onit_session` cookie (7-day lifetime) and
  restored across restarts, subject to the ownership rules above.

## Troubleshooting

**Startup error: "The web UI requires Google login, but no OAuth2
credentials are configured"**
Credentials are missing from all three sources. Run `onit setup` (or set
`GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`), and confirm with
`onit setup --show`. To run without login, pass `--no-login`.

**Google shows "Error 400: redirect_uri_mismatch"**
The exact callback URL (`http://<host>:<port>/auth/callback`) is not in the
OAuth client's **Authorized redirect URIs**. Add it character-for-character,
including the port. Each hostname you browse from needs its own entry.

**"Could not verify your Google account or you are not authorized" (403)**
The Google login succeeded but OnIt rejected the account. Causes:
- The account is not Google-hosted mail (e.g. a Google Account created on an
  `@outlook.com` address) — only Gmail and Workspace accounts are accepted.
- The email is not in `web_allowed_emails`, when that list is set.
- The email is unverified in Google.

**Google shows "Access blocked: … has not completed the Google verification
process"**
The OAuth consent screen is in *Testing* status and the account is not a
listed test user. Add it under **Audience → Test users**, or publish the app.

**Startup error: "Web UI login requires the google-auth package"**
```bash
pip install google-auth requests
```

## Production deployment

- **Use HTTPS.** Google requires `https` redirect URIs for non-localhost
  hosts. Put OnIt behind a TLS reverse proxy (nginx, Caddy):

  ```nginx
  server {
      listen 443 ssl;
      server_name your-domain.com;

      ssl_certificate /path/to/cert.pem;
      ssl_certificate_key /path/to/key.pem;

      location / {
          proxy_pass http://127.0.0.1:9000;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
      }
  }
  ```

  Then register `https://your-domain.com/auth/callback` as the redirect URI.
- **Restrict reachability** with firewall rules where possible:
  ```bash
  sudo ufw allow from 192.168.1.0/24 to any port 9000
  ```
- **Narrow the allowlist**: set `web_allowed_emails` to known users or your
  domain rather than relying on the Gmail/Workspace gate alone.
- **Never commit the client secret** — keep it in the keychain
  (`onit setup`) or environment variables.

## Disabling authentication

For local development on a trusted network only:

```bash
onit serve web --no-login
```

or in the config:

```yaml
web_require_auth: false
```

Anyone who can reach the port can then use the agent, read and upload files,
and see all chat sessions. Do not expose an unauthenticated instance beyond
localhost or a trusted LAN.

---

References:
- [Google OAuth2 documentation](https://developers.google.com/identity/protocols/oauth2)
- [OAuth 2.0 for Web Server Applications](https://developers.google.com/identity/protocols/oauth2/web-server)
