# Gateway Quick Start Guide

Chat with OnIt remotely from **Telegram** or **Viber** using a bot. This guide walks you through creating bot credentials and running the gateway from the CLI.

---

## Installation

Install OnIt with gateway support:

```bash
pip install "onit[gateway]"
# or, for everything:
pip install "onit[all]"
```

---

## Telegram Gateway

### Step 1: Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Follow the prompts — choose a **name** (e.g. "My OnIt Bot") and a **username** (must end in `bot`, e.g. `my_onit_bot`)
4. BotFather will reply with your **bot token**:
   ```
   Use this token to access the HTTP API:
   7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
5. Copy the token — you will need it in the next step

### Step 2: Set the Environment Variable

```bash
export TELEGRAM_BOT_TOKEN="7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Step 3: Run the Gateway

```bash
onit --gateway telegram
```

Or simply (Telegram is the default when `TELEGRAM_BOT_TOKEN` is set):

```bash
onit --gateway
```

You should see:

```
Telegram gateway running (Ctrl+C to stop)
```

### Step 4: Chat with Your Bot

1. Open Telegram on your phone or desktop
2. Search for your bot by its username (e.g. `@my_onit_bot`)
3. Tap **Start** and send a message
4. Send a photo to use vision capabilities (if your model supports it)

### Optional: Show Logs

To see messages and replies in your terminal:

```bash
onit --gateway telegram --gateway-show-logs
```

---

## Viber Gateway

### Step 1: Create a Viber Bot

> **Note:** Since February 2024, Viber bots can only be created on commercial terms. Visit [Viber for Business](https://www.viber.com/business/) to apply.

1. Go to the [Viber Admin Panel](https://partners.viber.com/) and sign in
2. Create a new bot account
3. In the bot's **Edit Info** screen, find your **Authentication Token**:
   ```
   4xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx=
   ```
4. Copy the token

### Step 2: Set Up a Public HTTPS URL

Viber requires a **publicly accessible HTTPS URL** to send webhook events to your server. Options:

**Option A — Cloudflare Tunnel (free, no account required for quick tunnels):**

```bash
# Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
# macOS: brew install cloudflared
cloudflared tunnel --url http://localhost:8443
```

Outputs a public URL like `https://abcd-1234-example.trycloudflare.com`. Use it with `/viber` appended as your webhook URL.

**Option B — ngrok (free tier available, requires account):**

```bash
# Install: https://ngrok.com/download
# macOS: brew install ngrok
# Then: ngrok config add-authtoken <your-token>
ngrok http 8443
```

Outputs a public URL like `https://abcd1234.ngrok-free.app`. Use it with `/viber` appended.

**Option C — localtunnel (free, no account required):**

```bash
# Install: npm install -g localtunnel
lt --port 8443
```

Outputs a public URL like `https://abcd-1234.loca.lt`. Use it with `/viber` appended. Note: clients may see a click-through interstitial on first access.

**Option D — Tailscale Funnel (free, requires Tailscale account):**

```bash
# Install: https://tailscale.com/download
# macOS: brew install tailscale
tailscale funnel 8443
```

Outputs a stable public URL like `https://your-machine.tail1234.ts.net`. Use it with `/viber` appended. Provides a persistent hostname tied to your machine.

**Option E — SSH reverse tunnel (any remote server with SSH):**

```bash
# Requires a VPS/server with a public IP and a reverse proxy (nginx/Caddy) handling HTTPS
ssh -R 8443:localhost:8443 user@your-server.com
```

Forward traffic from your server's port to your local machine. Pair with a reverse proxy on the server to terminate HTTPS.

**Option F — Production server:**

If you have a domain with HTTPS (e.g. behind nginx or Caddy), point it to the port OnIt listens on (default `8443`). See [DEPLOYMENT_WEB.md](DEPLOYMENT_WEB.md) for reverse proxy examples.

### Step 3: Set Environment Variables

```bash
export VIBER_BOT_TOKEN="4xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx="
export VIBER_WEBHOOK_URL="https://abcd-1234-example.trycloudflare.com/viber"
```

### Step 4: Run the Gateway

```bash
onit --gateway viber
```

You should see:

```
Viber gateway running on port 8443 (Ctrl+C to stop)
```

OnIt automatically registers the webhook URL with Viber on startup.

### Step 5: Chat with Your Bot

1. Open Viber on your phone
2. Search for your bot by name
3. Send a text message or a photo

### Optional: Custom Port and Logs

```bash
onit --gateway viber --viber-port 9090 --gateway-show-logs
```

Or pass the webhook URL via CLI instead of an environment variable:

```bash
onit --gateway viber --viber-webhook-url https://abcd-1234-example.trycloudflare.com/viber
```

---

## Auto-Detection

If you prefer not to specify `telegram` or `viber` explicitly, OnIt can auto-detect based on which environment variable is set:

```bash
# Will use Telegram if TELEGRAM_BOT_TOKEN is set,
# or Viber if VIBER_BOT_TOKEN is set
onit --gateway
```

When **both** tokens are set, Telegram is used by default. Use `--gateway viber` to override.

---

## CLI Reference

| Flag | Description |
|------|-------------|
| `--gateway` | Auto-detect gateway (Telegram or Viber) |
| `--gateway telegram` | Use Telegram gateway |
| `--gateway viber` | Use Viber gateway |
| `--gateway-show-logs` | Print messages and replies to terminal |
| `--viber-webhook-url URL` | Public HTTPS URL for Viber webhook |
| `--viber-port PORT` | Local port for Viber webhook server (default: 8443) |

| Environment Variable | Description |
|---------------------|-------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from BotFather |
| `VIBER_BOT_TOKEN` | Viber bot authentication token |
| `VIBER_WEBHOOK_URL` | Public HTTPS webhook URL for Viber |

---

## Troubleshooting

### Telegram: "Error: --gateway requires TELEGRAM_BOT_TOKEN"

**Cause:** The environment variable is not set or not exported.

**Solution:**
```bash
export TELEGRAM_BOT_TOKEN="your-token-here"
# Verify it's set:
echo $TELEGRAM_BOT_TOKEN
```

### Viber: "Failed to register Viber webhook"

**Cause:** The webhook URL is not reachable by Viber's servers, or is not HTTPS.

**Solution:**
- Ensure the URL uses `https://` (not `http://`)
- If using `cloudflared tunnel`, make sure it is running and the URL is current
- Check that your firewall allows inbound connections on the webhook port
- Verify the URL ends with `/viber` (e.g. `https://your-domain.com/viber`)

### Viber: Bot doesn't respond to messages

**Cause:** Webhook not registered or port mismatch.

**Solution:**
1. Check that `--viber-port` matches the port cloudflared/reverse proxy forwards to (default: 8443)
2. Restart the gateway — it re-registers the webhook on startup
3. Use `--gateway-show-logs` to confirm messages are being received

### Both: Model errors or slow responses

**Cause:** The underlying LLM host may be unavailable or overloaded.

**Solution:**
- Verify `--host` or `ONIT_HOST` points to a running LLM server
- Check the terminal for error messages
- Use `--gateway-show-logs` to see if messages arrive but responses fail
