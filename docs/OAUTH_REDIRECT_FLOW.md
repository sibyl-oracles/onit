# Full OAuth2 Redirect Flow - One-Click Google Sign-In

This guide covers the **complete OAuth2 redirect flow** implementation, which provides a seamless "Sign in with Google" experience - just like you see on other websites.

## What's Different?

### Old Method (Manual Token)
❌ Multi-step process
❌ Copy/paste tokens manually
❌ Tokens expire in 1 hour
❌ Requires OAuth Playground knowledge

### New Method (OAuth2 Redirect Flow) ✅
✅ **One-click sign-in**
✅ Automatic token handling
✅ **24-hour sessions**
✅ Standard OAuth flow (just like Gmail, GitHub, etc.)
✅ Shows Google account selector

## Setup Guide

### Step 1: Configure Google Cloud OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create or select your project
3. Click **"Create Credentials"** → **"OAuth client ID"**
4. Configure the OAuth consent screen if needed:
   - User Type: **External**
   - App name: `OnIt Web UI`
   - Add your email addresses
   - Scopes: `email`, `profile`, `openid`
5. For Application type, select: **Web application**
6. Add **Authorized redirect URIs** (⚠️ CRITICAL):
   ```
   http://localhost:9000/auth/callback
   http://YOUR_SERVER_IP:9000/auth/callback
   ```
   **Note:** Do NOT add `/oauthplayground` - that was only for the old method

7. Click **Create**
8. Copy BOTH credentials:
   - ✅ **Client ID**: `123456-abc.apps.googleusercontent.com`
   - ✅ **Client Secret**: `GOCSPX-xxxxxxxxxxxxx`

### Step 2: Configure OnIt

#### Option A: Configuration File

Edit `configs/web_oauth_example.yaml`:

```yaml
web: true
web_port: 9000

# OAuth2 Configuration
web_google_client_id: "YOUR_ACTUAL_CLIENT_ID.apps.googleusercontent.com"
web_google_client_secret: "GOCSPX-your-actual-secret-here"

# Optional: Restrict to specific emails or domains
# Supports exact emails and domain wildcards
web_allowed_emails:
  - "your-email@gmail.com"     # Exact email match
  - "*@company.com"             # All users from company.com domain
```

#### Option B: Command Line

```bash
onit --web \
  --google-client-id "YOUR_CLIENT_ID.apps.googleusercontent.com" \
  --google-client-secret "GOCSPX-xxxxx" \
  --allowed-emails "your-email@gmail.com"
```

### Step 3: Run OnIt

```bash
onit --config configs/web_oauth_example.yaml
```

You should see:
```
🔐 Google Client ID: SET
🔐 Google Client Secret: SET
🔐 OAuth2 Redirect Flow ENABLED
   Client ID: 613088846860-hule...
   Allowed emails: your-email@gmail.com
============================================================
🚀 Launching OnIt Web UI on http://0.0.0.0:9000
🔐 OAuth2 Authentication: ENABLED
   Login URL: http://localhost:9000/auth/login
   ⚠️  Login required before accessing chat interface
============================================================
```

## How to Use

### 1. Access the Web UI

Navigate to: `http://localhost:9000` (or your server's IP)

### 2. Click "Sign in with Google"

You'll see a beautiful login page with a Google sign-in button.

### 3. Choose Your Google Account

Google will show you the account selector - pick the account you want to use.

### 4. Grant Permissions

Google will ask you to authorize OnIt to:
- View your email address
- View your basic profile info

Click **"Allow"**

### 5. You're In! 🎉

You'll be automatically redirected back to the chat interface, fully authenticated.

Your session lasts **24 hours** - no need to sign in again during that time!

## API Endpoints

The OAuth2 flow adds these endpoints:

### `GET /auth/login`
Initiates the OAuth flow. Redirects to Google's account selector.

### `GET /auth/callback`
Handles the OAuth callback from Google. Validates the authorization code and creates a session.

### `GET /auth/logout`
Logs out the current user and clears the session.

### `GET /auth/check`
Checks if the current user is authenticated. Returns:
```json
{
  "authenticated": true,
  "email": "user@gmail.com"
}
```

## Security Features

### 1. PKCE (Proof Key for Code Exchange)
- Prevents authorization code interception attacks
- Uses SHA-256 hashed code challenges
- Code verifier stored server-side only

### 2. State Parameter
- CSRF protection
- Random 32-byte tokens
- One-time use only
- Expires after 10 minutes

### 3. Secure Session Cookies
- HttpOnly flag (prevents XSS)
- SameSite=Lax (prevents CSRF)
- 24-hour expiration
- Stored server-side

### 4. Token Verification
- Google ID tokens verified against Google's public keys
- Email verification required
- Optional email whitelist

## Troubleshooting

### "Client Secret not configured" Error

**Problem:** You set `web_google_client_id` but not `web_google_client_secret`

**Solution:** Both credentials are required for OAuth redirect flow:
```yaml
web_google_client_id: "YOUR_CLIENT_ID"
web_google_client_secret: "YOUR_CLIENT_SECRET"
```

### "redirect_uri_mismatch" Error

**Problem:** The redirect URI doesn't match what's configured in Google Cloud Console

**Solution:**
1. Check Google Cloud Console → Your OAuth Client → Authorized redirect URIs
2. Must exactly match: `http://localhost:9000/auth/callback` (or your server's address)
3. No trailing slashes, exact port number

### "Invalid or Expired Session" Error

**Problem:** The OAuth flow took too long (>10 minutes) or was already used

**Solution:** Simply try logging in again - start fresh

### Authentication Works But Chat Doesn't Load

**Problem:** Session cookie not being sent/received properly

**Solution:**
- Make sure you're accessing via the same hostname used in the redirect URI
- Check browser console for errors
- Try clearing cookies and logging in again

### "Invalid Grant" Error

**Problem:** Authorization code has already been used or is expired

**Solution:**
- Authorization codes are one-time use
- Start the login flow again from `/auth/login`

## Comparison: Old vs New Method

| Feature | Manual Token Method | OAuth2 Redirect Flow |
|---------|-------------------|----------------------|
| User clicks | 10+ clicks | 2 clicks |
| Steps | 9 steps | 2 steps |
| External sites | OAuth Playground | None |
| Copy/paste | Required | Not needed |
| Session duration | 1 hour | 24 hours |
| Setup complexity | Medium | Medium |
| Security | Secure | More secure (PKCE) |
| User experience | Poor | Excellent |

## For Developers

### Flow Diagram

```
┌─────────┐                                        ┌─────────┐
│ Browser │                                        │  OnIt   │
│         │                                        │ Server  │
└────┬────┘                                        └────┬────┘
     │                                                  │
     │  1. GET /                                        │
     ├─────────────────────────────────────────────────>│
     │                                                  │
     │  2. Login page (if not authenticated)           │
     │<─────────────────────────────────────────────────┤
     │                                                  │
     │  3. Click "Sign in with Google"                 │
     │  GET /auth/login                                 │
     ├─────────────────────────────────────────────────>│
     │                                                  │
     │  4. Create PKCE params, state                    │
     │     Redirect to Google                           │
     │<─────────────────────────────────────────────────┤
     │                                                  │
     ├──────────────────────┐                          │
     │   5. Google OAuth    │                          │
     │   Account Selector   │                          │
     │   User Authorizes    │                          │
     │<─────────────────────┘                          │
     │                                                  │
     │  6. GET /auth/callback?code=xxx&state=yyy       │
     ├─────────────────────────────────────────────────>│
     │                                                  │
     │                              7. Verify state,    │
     │                              Exchange code for   │
     │                              token with PKCE,    │
     │                              Create session      │
     │                                                  │
     │  8. Set cookie, redirect to /                   │
     │<─────────────────────────────────────────────────┤
     │                                                  │
     │  9. GET / (with auth cookie)                    │
     ├─────────────────────────────────────────────────>│
     │                                                  │
     │  10. Chat interface ✅                          │
     │<─────────────────────────────────────────────────┤
     │                                                  │
```

### Code Snippet: Custom OAuth Handler

If you want to customize the OAuth flow, you can modify the `_setup_oauth_routes` method in `onit/ui/web.py`.

Example - Add custom success page:

```python
@fastapi_app.get("/auth/callback")
async def oauth_callback(request: Request, response: Response):
    # ... existing code ...

    # Custom success page instead of redirect
    return HTMLResponse(f"""
        <html>
        <head>
            <title>Success!</title>
            <meta http-equiv="refresh" content="2;url=/" />
        </head>
        <body style="text-align: center; padding: 50px;">
            <h1>✅ Welcome, {email}!</h1>
            <p>Redirecting to chat...</p>
        </body>
        </html>
    """)
```

---

## Summary

The OAuth2 redirect flow provides a **production-ready**, **user-friendly** authentication system for OnIt's web UI. It's secure, standards-compliant, and provides an experience users expect from modern web applications.

**Next Steps:**
1. Set up your Google OAuth credentials
2. Add the redirect URI: `http://localhost:9000/auth/callback`
3. Configure both Client ID and Client Secret
4. Launch and enjoy one-click sign-in! 🚀
