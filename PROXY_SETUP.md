# Proxy Setup Guide for Poly-Maker Bot

This guide explains how to run the bot on your local machine while using a remote server as a proxy.

## Architecture

```
[Local Machine]  -->  [SSH Tunnel/SOCKS Proxy]  -->  [Remote Server]  -->  [Polymarket/Polygon]
   (Bot runs)           (Encrypted tunnel)              (Proxy only)           (APIs)
```

## Benefits

1. **More Resources**: Run bot on your powerful local machine instead of limited VPS
2. **Better Monitoring**: Access to local logs, debugging tools, and UI
3. **Cost Effective**: Keep cheap VPS as proxy, run heavy bot locally
4. **No OOM Issues**: Use your local machine's full memory

---

## Option 1: SSH SOCKS Proxy (Recommended)

### Step 1: Create SSH Tunnel from Local Machine

On your **local machine**, run:

```bash
# Create SOCKS proxy on port 1080 via SSH tunnel
ssh -D 1080 -C -N root@YOUR_SERVER_IP

# Or run in background with autossh (more reliable)
autossh -M 0 -f -D 1080 -C -N root@YOUR_SERVER_IP
```

**Explanation:**
- `-D 1080`: Create SOCKS5 proxy on localhost:1080
- `-C`: Enable compression
- `-N`: Don't execute remote command (just tunnel)
- `-f`: Run in background (autossh only)
- `-M 0`: Disable autossh monitoring port (autossh only)

### Step 2: Configure Bot to Use SOCKS Proxy

Add to your **local `.env` file**:

```bash
# SOCKS Proxy Configuration
SOCKS_PROXY_URL=socks5://localhost:1080
```

### Step 3: Install SOCKS Support

The bot requires `requests[socks]` for SOCKS proxy support:

```bash
# On local machine
uv pip install "requests[socks]" PySocks
```

### Step 4: Run Bot Locally

```bash
cd /path/to/poly-maker
uv run python main.py
```

You should see:
```
============================================================
🔧 Configuring Proxy Support
============================================================
✓ Proxy detected:
  HTTP: socks5://localhost:1080
  HTTPS: socks5://localhost:1080
✓ Patched py_clob_client to use proxy
✓ All HTTP traffic will be routed through proxy
============================================================
```

---

## Option 2: HTTP Proxy with nginx/squid

If you prefer HTTP proxy instead of SOCKS:

### Step 1: Install Squid on Remote Server

```bash
# On remote server
apt update
apt install squid -y

# Configure squid
cat > /etc/squid/squid.conf <<EOF
# Allow only from your local IP (replace with your IP)
acl localnet src YOUR_LOCAL_IP/32

# Squid listening port
http_port 3128

# Access control
http_access allow localnet
http_access deny all

# Disable caching (we just want proxying)
cache deny all
EOF

# Restart squid
systemctl restart squid
systemctl enable squid

# Open firewall (if using ufw)
ufw allow from YOUR_LOCAL_IP to any port 3128
```

### Step 2: Configure Bot

Add to local `.env`:

```bash
# HTTP Proxy Configuration
HTTP_PROXY=http://YOUR_SERVER_IP:3128
HTTPS_PROXY=http://YOUR_SERVER_IP:3128
```

---

## Option 3: SSH Port Forwarding (Alternative)

Forward individual ports instead of SOCKS proxy:

```bash
# Forward Polygon RPC (443) and Polymarket API (443)
ssh -L 8545:polygon-rpc.com:443 \
    -L 8546:clob.polymarket.com:443 \
    -N root@YOUR_SERVER_IP
```

Then update bot code to use `localhost:8545` and `localhost:8546`.

---

## Verification

### Test Proxy Connection

On **local machine**:

```bash
# Test SOCKS proxy
curl --socks5 localhost:1080 https://api.ipify.org?format=json

# Should return your SERVER's IP, not your local IP
```

### Test Bot Proxy

```bash
# Run bot with verbose proxy logging
uv run python -c "
from poly_utils.proxy_config import setup_proxy, get_proxy_session

# Setup proxy
setup_proxy(verbose=True)

# Test request
session = get_proxy_session()
response = session.get('https://api.ipify.org?format=json')
print(f'\\nRequest IP: {response.json()}')
print('If this shows your server IP, proxy is working!')
"
```

---

## Security Considerations

### 1. Secure Your SSH Tunnel

```bash
# Use SSH key authentication only
# In remote server's /etc/ssh/sshd_config:
PasswordAuthentication no
PubkeyAuthentication yes

# Restart SSH
systemctl restart sshd
```

### 2. Restrict Proxy Access

For Squid, only allow your local IP:
```bash
acl localnet src YOUR_LOCAL_IP/32
http_access allow localnet
http_access deny all
```

For SSH, use `GatewayPorts no` in `/etc/ssh/sshd_config`

### 3. Monitor Proxy Usage

```bash
# Monitor SSH tunnel
watch 'ss -tnp | grep :1080'

# Monitor Squid
tail -f /var/log/squid/access.log
```

---

## Troubleshooting

### Issue: "Connection refused" or "Proxy error"

**Solution:**
1. Check SSH tunnel is running: `ps aux | grep "ssh -D"`
2. Test tunnel: `curl --socks5 localhost:1080 https://google.com`
3. Check firewall allows SSH: `ufw status`

### Issue: "Failed to patch py_clob_client"

**Solution:**
1. Ensure SOCKS support: `uv pip install "requests[socks]" PySocks`
2. Check proxy URL format: Must be `socks5://localhost:1080`

### Issue: Bot connects but no orders placed

**Solution:**
1. Check remote server can reach Polymarket: `curl https://clob.polymarket.com`
2. Verify credentials in `.env` are correct
3. Check logs for specific errors

### Issue: "Unable to import module PySocks"

**Solution:**
```bash
uv pip install PySocks requests[socks]
```

---

## Performance Optimization

### Keep Tunnel Alive

Create `~/.ssh/config` on local machine:

```
Host proxy-server
    HostName YOUR_SERVER_IP
    User root
    ServerAliveInterval 60
    ServerAliveCountMax 3
    DynamicForward 1080
    Compression yes
```

Then connect with: `ssh -N proxy-server`

### Use autossh for Auto-Reconnect

```bash
# Install autossh
apt install autossh  # Ubuntu/Debian
brew install autossh  # macOS

# Run with auto-reconnect
autossh -M 0 -f -N proxy-server
```

### Create Systemd Service (Linux)

Create `/etc/systemd/system/socks-proxy.service`:

```ini
[Unit]
Description=SOCKS Proxy Tunnel
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/usr/bin/ssh -D 1080 -C -N root@YOUR_SERVER_IP
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
systemctl daemon-reload
systemctl enable socks-proxy
systemctl start socks-proxy
```

---

## Migration Checklist

- [ ] Setup SSH tunnel on local machine
- [ ] Test tunnel with curl
- [ ] Install PySocks: `uv pip install "requests[socks]" PySocks`
- [ ] Copy `.env` file from server to local machine
- [ ] Add `SOCKS_PROXY_URL=socks5://localhost:1080` to local `.env`
- [ ] Clone repo to local machine
- [ ] Install dependencies: `uv sync`
- [ ] Test proxy: Run verification script above
- [ ] Run bot locally: `uv run python main.py`
- [ ] Verify bot connects and trades
- [ ] Stop bot on remote server
- [ ] Setup autossh for reliable tunnel
- [ ] (Optional) Create systemd service for tunnel

---

## Environment Variables Reference

```bash
# Option 1: SOCKS Proxy (Recommended)
SOCKS_PROXY_URL=socks5://localhost:1080

# Option 2: Single proxy for all traffic
PROXY_URL=http://YOUR_SERVER_IP:3128

# Option 3: Separate HTTP/HTTPS proxies
HTTP_PROXY=http://YOUR_SERVER_IP:3128
HTTPS_PROXY=http://YOUR_SERVER_IP:3128

# Optional: Bypass proxy for specific hosts
NO_PROXY=localhost,127.0.0.1
```

---

## Questions?

- Check bot logs for detailed proxy connection info
- Run `setup_proxy(verbose=True)` to see proxy configuration
- Test with minimal script before running full bot
