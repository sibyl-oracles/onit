# Web UI Google OAuth2 Authentication

This document explains how to set up Google OAuth2 authentication for the OnIt web interface, securing it from unauthorized access when exposed to external networks.

## Why Authentication is Important

When running the web server with `server_name="0.0.0.0"`, it becomes accessible from any network interface, including external IP addresses. Without authentication:
- Anyone who can reach your server can use the OnIt agent
- Uploaded files are accessible to all users
- No session isolation between users
- Potential security and privacy risks

## Setting Up Google OAuth2

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter a project name (e.g., "OnIt Web Auth")
4. Click "Create"

### Step 2: Enable Required APIs

1. In the Google Cloud Console, navigate to "APIs & Services" → "Library"
2. Search for "Google+ API" and enable it
3. This allows your application to verify user identities

### Step 3: Create OAuth 2.0 Credentials

1. Navigate to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - User Type: "External" (or "Internal" for Google Workspace)
   - Fill in required fields (App name, user support email, developer email)
   - Add scopes: `email` and `profile`
   - Add test users if using external user type
4. For Application type, select "Web application"
5. Add a name (e.g., "OnIt Web Client")
6. Under "Authorized JavaScript origins", add:
   - `http://localhost:9000` (for local development)
   - `http://YOUR_SERVER_IP:9000` (for production)
   - `https://YOUR_DOMAIN` (if using HTTPS)
7. Under "Authorized redirect URIs", add:
   - `https://developers.google.com/oauthplayground` (for OAuth Playground)
8. Click "Create"
9. **Important:** Copy both:
   - **Client ID** (looks like: `123456789-abcdefg.apps.googleusercontent.com`)
   - **Client Secret** (looks like: `GOCSPX-xxxxxxxxxxxxx`)

   You'll need both for the OAuth Playground to generate tokens.

### Step 4: Configure OnIt

#### Option A: Using Configuration File

1. Copy the example configuration:
   ```bash
   cp configs/web_oauth_example.yaml configs/my_config.yaml
   ```

2. Edit `configs/my_config.yaml` and replace:
   ```yaml
   web_google_client_id: "YOUR_ACTUAL_CLIENT_ID.apps.googleusercontent.com"
   ```

3. (Optional) Restrict to specific email addresses:
   ```yaml
   web_allowed_emails:
     - "your-email@gmail.com"
     - "colleague@company.com"
   ```

4. Run OnIt:
   ```bash
   onit --config configs/my_config.yaml
   ```

#### Option B: Using Command Line Arguments

```bash
onit --web \
  --google-client-id "YOUR_CLIENT_ID.apps.googleusercontent.com" \
  --allowed-emails "email1@gmail.com,email2@gmail.com"
```

### Step 5: Using the Authenticated Web Interface

1. Start the OnIt server with authentication enabled
2. Open your browser and navigate to `http://localhost:9000` (or your server URL)
3. You'll see instructions on how to get your Google ID token
4. Follow the instructions to obtain your token from Google OAuth Playground
5. Paste the token into the input field
6. Click "Authenticate"
7. After successful authentication, you'll see the chat interface

#### How to Get Your Google ID Token:

1. Visit [Google OAuth Playground](https://developers.google.com/oauthplayground/)
2. Click the ⚙️ gear icon (OAuth 2.0 configuration)
3. Check "Use your own OAuth credentials"
4. Enter your **Client ID** and **Client Secret** from Step 3
5. In the left panel, select **Google OAuth2 API v2** → `https://www.googleapis.com/auth/userinfo.email`
6. Click **"Authorize APIs"** and sign in with your Google account
7. Click **"Exchange authorization code for tokens"**
8. Copy the **id_token** value (long string starting with "eyJ...")
9. Paste it into the OnIt login page and click "Authenticate"

**Note:** ID tokens expire after 1 hour. If you get an authentication error, generate a new token.

## Security Features

### Session Management
- Sessions are valid for 24 hours by default
- Sessions are stored in memory (cleared on server restart)
- Each user has an isolated session

### Token Verification
- Google ID tokens are verified against Google's public keys
- Only tokens from your configured Client ID are accepted
- Email addresses must be verified by Google

### Access Control
- If `web_allowed_emails` is configured, only listed emails can access
- If not configured, any verified Google account can access
- Unauthorized attempts are logged and rejected

## Troubleshooting

### "Authentication failed. Invalid token or unauthorized email"

**Causes:**
- Invalid or expired token
- Wrong Client ID
- Email not in `web_allowed_emails` list
- Email not verified by Google

**Solutions:**
- Verify your Client ID is correct
- Ensure your email is verified in Google
- Check `web_allowed_emails` configuration
- Try signing in again to get a fresh token

### "Invalid token" or token expired

**Causes:**
- Token has expired (tokens are valid for 1 hour)
- Invalid token format
- Token was copied incorrectly

**Solutions:**
- Generate a new token from Google OAuth Playground
- Ensure you copy the entire `id_token` value
- Make sure there are no extra spaces or line breaks

### "google-auth library not found"

**Solution:**
Install the required dependency:
```bash
pip install google-auth
```

## Best Practices

1. **Keep Client ID Secure**: Don't commit it to public repositories
2. **Use Environment Variables**: Store sensitive config in environment variables
   ```bash
   export GOOGLE_CLIENT_ID="your-client-id"
   ```
3. **HTTPS in Production**: Use a reverse proxy (nginx, Caddy) with SSL/TLS
4. **Restrict Email List**: Limit access to known users
5. **Regular Audits**: Review Google Cloud Console audit logs
6. **Session Timeout**: Consider implementing shorter session durations for sensitive environments

## Additional Security Measures

While OAuth2 provides authentication, consider these additional security layers:

### 1. Reverse Proxy with HTTPS
Use nginx or Caddy to add HTTPS:

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

### 2. Firewall Rules
Restrict access by IP:
```bash
# Allow only specific IP range
sudo ufw allow from 192.168.1.0/24 to any port 9000
```

### 3. Rate Limiting
Consider implementing rate limiting at the reverse proxy level to prevent abuse.

## Disabling Authentication

To run without authentication (NOT recommended for external access):

```bash
# Simply don't provide google-client-id
onit --web
```

Or in config:
```yaml
web: true
web_port: 9000
# Don't set web_google_client_id
```

---

For more information, see:
- [Google OAuth2 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [Google Sign-In for Websites](https://developers.google.com/identity/gsi/web)
