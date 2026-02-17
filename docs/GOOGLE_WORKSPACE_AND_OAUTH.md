# Google Workspace and OAuth for Gmail

This tutorial walks through setting up Google Workspace integration (Drive, Docs, Sheets, Slides, Gmail) and OAuth authentication for the OnIt web UI.

OnIt uses **two separate authentication systems**:

| System | Purpose | Credential type |
|--------|---------|-----------------|
| **Google Workspace MCP Server** | Agent reads/writes Drive, Docs, Sheets, Slides, Gmail | Service account JSON key |
| **Web UI OAuth** | Users sign in to the browser interface with their Google account | OAuth 2.0 client ID + secret |

You can enable one or both depending on your needs.

---

## Part 1 — Google Workspace MCP Server

The Google Workspace MCP server gives OnIt 22 tools across Drive, Docs, Sheets, Slides, and Gmail. It authenticates using a **service account**, not a personal Google login.

### 1.1 Create a Google Cloud project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a project** > **New Project**
3. Name it (e.g. `onit-workspace`) and click **Create**

### 1.2 Enable APIs

In your project, go to **APIs & Services** > **Library** and enable all of the following:

- Google Docs API
- Google Sheets API
- Google Slides API
- Google Drive API
- Gmail API

### 1.3 Create a service account

1. Go to **IAM & Admin** > **Service Accounts**
2. Click **Create Service Account**
3. Name it (e.g. `onit-agent`) and click **Create and Continue**
4. Skip the optional role and user access steps — click **Done**
5. Click on the service account you just created
6. Go to the **Keys** tab > **Add Key** > **Create new key** > **JSON**
7. Save the downloaded JSON file (e.g. `~/.config/gcloud/credentials.json`)

### 1.4 Configure OnIt to use the service account

There are three ways to provide credentials (checked in this order):

**Option A — MCP server config:**

Edit `src/mcp/servers/configs/default.yaml`:

```yaml
- name: GoogleWorkspaceMCPServer
  module: tasks.office.google
  # ...
  credentials_file: ~/.config/gcloud/credentials.json
```

**Option B — Environment variable:**

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/credentials.json
```

**Option C — Default path:**

Place the JSON key at `~/.config/gcloud/credentials.json` — it will be found automatically.

### 1.5 Start the MCP servers

```bash
onit --mcp
```

Verify the Google Workspace server is running on port 9006. Then start the agent:

```bash
onit
```

Ask OnIt to run the `google_auth` tool to confirm authentication is working.

### 1.6 Share files with the service account

Service accounts have their own Drive storage. To access **existing** files:

- Open the file in Google Drive
- Click **Share** and add the service account email (e.g. `onit-agent@onit-workspace.iam.gserviceaccount.com`)

Files **created** by the service account are owned by it and not visible to you unless explicitly shared via the `share_with` parameter in the tools.

### 1.7 Available tools

| Category | Tools |
|----------|-------|
| Auth | `google_auth` |
| Drive | `drive_list`, `drive_create_folder`, `drive_move`, `drive_delete`, `drive_share`, `drive_download`, `drive_upload` |
| Docs | `doc_create`, `doc_read`, `doc_write` |
| Sheets | `sheet_create`, `sheet_read`, `sheet_write` |
| Slides | `slides_create`, `slides_edit`, `slides_read` |
| Gmail | `gmail_list`, `gmail_read`, `gmail_modify`, `gmail_send`, `gmail_attachment`, `gmail_create_label` |

---

## Part 2 — Domain-Wide Delegation (required for Gmail)

Gmail tools **always** require domain-wide delegation because the Gmail API does not allow direct service account access to mailboxes. Delegation lets the service account impersonate a real user.

This also makes Drive/Docs/Sheets/Slides tools more practical — created files appear in the impersonated user's Drive instead of the service account's.

> **Note:** Domain-wide delegation requires a **Google Workspace** (organization) account. It does not work with personal `@gmail.com` accounts.

### 2.1 Enable delegation on the service account

1. Go to **IAM & Admin** > **Service Accounts**
2. Click on your service account
3. Check **Enable Google Workspace Domain-wide Delegation** (under the Details tab, or via the three-dot menu > **Edit**)
4. Note the **Client ID** (numeric, e.g. `1234567890`)

### 2.2 Authorize scopes in Google Admin Console

1. Go to [admin.google.com](https://admin.google.com/)
2. Navigate to **Security** > **Access and data control** > **API Controls** > **Domain-wide delegation**
3. Click **Add new**
4. Enter the service account **Client ID**
5. Add these **OAuth scopes** (comma-separated):

```
https://www.googleapis.com/auth/documents,
https://www.googleapis.com/auth/spreadsheets,
https://www.googleapis.com/auth/presentations,
https://www.googleapis.com/auth/drive,
https://www.googleapis.com/auth/gmail.modify,
https://www.googleapis.com/auth/gmail.send,
https://www.googleapis.com/auth/gmail.compose
```

6. Click **Authorize**

### 2.3 Configure the delegated user

You can set a default user to impersonate in the MCP server config:

```yaml
- name: GoogleWorkspaceMCPServer
  module: tasks.office.google
  credentials_file: ~/.config/gcloud/credentials.json
  delegated_user: user@yourcompany.com
```

Or pass `user_email` per-tool call. For example, when OnIt calls `gmail_send`, it will use `user_email` to send as that user.

### 2.4 Test Gmail

Start the MCP servers and agent, then ask OnIt:

> "List my recent emails" (with `user_email` set to a real user in your organization)

If delegation is configured correctly, you'll see the user's inbox.

---

## Part 3 — Web UI OAuth (Google Sign-In)

This lets users authenticate to the OnIt web UI using their Google account. It is completely independent from the Workspace MCP server.

### 3.1 Create OAuth 2.0 credentials

1. In your Google Cloud project, go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth client ID**
3. If prompted, configure the **OAuth consent screen**:
   - User Type: **External** (or Internal for Workspace orgs)
   - App name: `OnIt Web UI`
   - Add your email as a support and developer contact
   - Scopes: add `email`, `profile`, `openid`
   - Add test users if the app is in "Testing" mode
4. Application type: **Web application**
5. Name: `OnIt Web Client`
6. Under **Authorized redirect URIs**, add:

```
http://localhost:9000/auth/callback
```

If you'll access the UI from another machine, also add:

```
http://YOUR_SERVER_IP:9000/auth/callback
```

7. Click **Create**
8. Copy the **Client ID** and **Client Secret**

### 3.2 Configure OnIt

Edit your config file (e.g. copy `configs/default.yaml` or create a new one):

```yaml
web: true
web_port: 9000

# OAuth2 credentials
web_google_client_id: "YOUR_CLIENT_ID.apps.googleusercontent.com"
web_google_client_secret: "GOCSPX-your-secret-here"

# Optional: restrict access to specific emails or domains
web_allowed_emails:
  - "admin@yourcompany.com"       # exact email
  - "*@yourcompany.com"           # entire domain
```

> If `web_google_client_id` or `web_google_client_secret` is missing or contains `YOUR_`, authentication is disabled and the web UI is open to anyone.

### 3.3 Launch the web UI

```bash
onit --config configs/your_config.yaml
```

You should see:

```
OAuth2 authentication enabled (client: 123456789-abc...)
   Allowed emails: admin@yourcompany.com, *@yourcompany.com

============================================================
Launching OnIt Web UI on http://0.0.0.0:9000
   OAuth2 Authentication: ENABLED
   Login URL: http://localhost:9000/auth/login
============================================================
```

### 3.4 Sign in

1. Open `http://localhost:9000` in your browser
2. Click **Sign in with Google**
3. Select your Google account and grant permissions
4. You're redirected to the chat interface

Sessions last **24 hours**. Click **Logout** to end the session early.

### 3.5 Security features

The OAuth flow includes:

- **PKCE** (Proof Key for Code Exchange) — prevents authorization code interception
- **State parameter** — CSRF protection, one-time use, expires in 10 minutes
- **HttpOnly cookies** — prevents XSS token theft
- **SameSite=Lax** — prevents cross-site request forgery
- **Email allowlist** — restricts access to specified users or domains

---

## Part 4 — Putting It All Together

A typical production setup enables both systems. Here's a complete config:

```yaml
serving:
  host: https://openrouter.ai/api/v1
  host_key: sk-or-v1-your-key-here
  model: google/gemini-2.5-pro
  think: true
  max_tokens: 262144

# Web UI with OAuth
web: true
web_port: 9000
web_google_client_id: "YOUR_CLIENT_ID.apps.googleusercontent.com"
web_google_client_secret: "GOCSPX-your-secret"
web_allowed_emails:
  - "*@yourcompany.com"

# MCP servers (including Google Workspace)
mcp:
  servers:
    - name: PromptsMCPServer
      description: Prompt templates
      url: http://127.0.0.1:9002/prompts
      enabled: true

    - name: WebSearchHandler
      description: Web search
      url: http://127.0.0.1:9003/search
      enabled: true

    - name: BashMCPServer
      description: Bash commands
      url: http://127.0.0.1:9004/bash
      enabled: true

    - name: OfficeMCPServer
      description: Microsoft Office
      url: http://127.0.0.1:9005/office
      enabled: true

    - name: GoogleWorkspaceMCPServer
      description: Google Workspace
      url: http://127.0.0.1:9006/workspace
      enabled: true
```

Then in a separate terminal:

```bash
# Terminal 1: Start MCP servers (including Google Workspace on port 9006)
onit --mcp

# Terminal 2: Start the web UI with OAuth
onit --config configs/your_config.yaml
```

With Docker:

```bash
docker run -it --rm \
  -p 9000:9000 \
  -p 9001-9006:9001-9006 \
  -v ~/.config/gcloud/credentials.json:/app/credentials.json:ro \
  --env-file .env \
  onit --web --web-port 9000
```

---

## Troubleshooting

### "Authentication DISABLED" in web UI

Both `web_google_client_id` and `web_google_client_secret` must be set to real values (not placeholders containing `YOUR_`).

### "redirect_uri_mismatch" error

The redirect URI in Google Cloud Console must **exactly** match `http://HOST:PORT/auth/callback`. Check for trailing slashes, wrong ports, or `http` vs `https`.

### "permission_denied" from Google Workspace tools

- Verify the APIs are enabled in Google Cloud Console
- For personal files: share them with the service account email
- For Gmail: domain-wide delegation is required (see Part 2)

### "storage quota" error on Drive upload

Service accounts have no storage quota. Either:
- Use `user_email` to impersonate a user with quota (requires delegation)
- Upload to a Shared Drive folder

### Gmail tools return 403

- Gmail **always** requires domain-wide delegation — it cannot be used with a service account alone
- Verify the Gmail scopes are authorized in Google Admin Console
- Check that `user_email` is a valid user in your Workspace organization

### "No credentials found"

Ensure one of these exists:
1. `credentials_file` in MCP server config
2. `GOOGLE_APPLICATION_CREDENTIALS` environment variable
3. JSON key at `~/.config/gcloud/credentials.json`

---

## Further reading

- [OAUTH_REDIRECT_FLOW.md](OAUTH_REDIRECT_FLOW.md) — detailed OAuth2 redirect flow implementation
- [OAUTH_SETUP_QUICK_START.md](OAUTH_SETUP_QUICK_START.md) — quick OAuth setup checklist
- [WEB_AUTHENTICATION.md](WEB_AUTHENTICATION.md) — web authentication details
