# Google OAuth Quick Setup Guide

## Problem: Login Prompt Not Showing

If you're running the web server and don't see a login prompt, it means authentication is **disabled**. This usually happens when:

1. ❌ No `web_google_client_id` configured
2. ❌ Client ID is still set to the placeholder value `YOUR_GOOGLE_CLIENT_ID_HERE`
3. ❌ Client ID is invalid or not properly loaded

## Quick Fix (5 minutes)

### Step 1: Get Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project (or select existing)
3. Click **"Create Credentials"** → **"OAuth client ID"**
4. If needed, configure consent screen:
   - User Type: **External**
   - App name: `OnIt Web UI`
   - Add your email
   - Add scope: `userinfo.email`
   - Add test users if needed
5. Application type: **Web application**
6. Name: `OnIt Web Client`
7. **Authorized redirect URIs**: Add `https://developers.google.com/oauthplayground`
8. Click **Create**
9. **Copy both**:
   - ✅ Client ID (e.g., `123456-abc.apps.googleusercontent.com`)
   - ✅ Client Secret (e.g., `GOCSPX-xxxxx`)

### Step 2: Update Your Config

Edit your config file (e.g., `configs/web_oauth_example.yaml`):

```yaml
# Replace this placeholder:
web_google_client_id: "YOUR_GOOGLE_CLIENT_ID_HERE.apps.googleusercontent.com"

# With your actual Client ID:
web_google_client_id: "123456789-abcdefghijk.apps.googleusercontent.com"

# Optional: Restrict to specific emails
web_allowed_emails:
  - "your-email@gmail.com"
  - "colleague@company.com"
```

### Step 3: Run OnIt

```bash
onit --config configs/web_oauth_example.yaml
```

You should see:
```
🔐 Authentication ENABLED with client ID: 123456789-abcdefghi...
   Allowed emails: your-email@gmail.com
============================================================
🚀 Launching OnIt Web UI on http://0.0.0.0:9000
🔐 Authentication: ENABLED
   ⚠️  Login required before accessing chat interface
============================================================
```

### Step 4: Get Your Login Token

1. Visit [Google OAuth Playground](https://developers.google.com/oauthplayground/)
2. Click ⚙️ → "Use your own OAuth credentials"
3. Enter your **Client ID** and **Client Secret**
4. Select: `Google OAuth2 API v2` → `userinfo.email`
5. Click **"Authorize APIs"** → Sign in with Google
6. Click **"Exchange authorization code for tokens"**
7. Copy the **id_token** (long string starting with "eyJ...")

### Step 5: Login

1. Go to `http://localhost:9000` (or your server IP)
2. You should see: **🔐 OnIt Chat - Authentication Required**
3. Paste your token
4. Click **"Authenticate"**
5. You're in! 🎉

## Troubleshooting

### "Authentication DISABLED" message

**Cause:** Client ID is not configured or is a placeholder.

**Fix:** Replace the placeholder with your actual Google Client ID in the config file.

### "Authentication failed. Invalid token or unauthorized email"

**Causes:**
- Token expired (they last 1 hour)
- Email not in `web_allowed_emails` list
- Email not verified by Google

**Fix:**
- Generate a new token
- Check your email is in the allowed list
- Verify your email in Google account settings

### Config file not being read

**Check:**
```bash
# Make sure you're using the right config file
onit --config configs/web_oauth_example.yaml

# Or set it in the default config
cp configs/web_oauth_example.yaml configs/default.yaml
onit
```

### Still seeing "Session expired"

**Cause:** Session validation is working (this is good!) but your session expired.

**Fix:** Tokens expire after 1 hour. Generate a new token and login again.

## Alternative: Run Without Authentication (NOT RECOMMENDED)

If you only need local access and don't want authentication:

```bash
# Don't set web_google_client_id
onit --web
```

⚠️ **WARNING:** This makes the server accessible to anyone on your network!

## Security Reminders

✅ **DO:**
- Use authentication when binding to 0.0.0.0
- Keep Client Secret secure
- Use HTTPS in production
- Restrict `web_allowed_emails` to known users

❌ **DON'T:**
- Commit Client ID/Secret to git
- Share your tokens
- Run without auth on public networks
- Use placeholder values in production

---

Need more help? See [WEB_AUTHENTICATION.md](WEB_AUTHENTICATION.md) for detailed instructions.
