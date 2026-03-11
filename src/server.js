/**
 * OpenClaw gateway wrapper for Stratum Sourcing Monitor.
 * Proxies requests to the OpenClaw gateway and serves setup UI.
 * Based on the openclaw-railway-template pattern.
 */

const http = require('http');
const { spawn } = require('child_process');

const PORT = process.env.PORT || 8080;
const INTERNAL_PORT = process.env.OPENCLAW_INTERNAL_PORT || 9080;

let gatewayProcess = null;
let gatewayReady = false;

function startGateway() {
  const stateDir = process.env.OPENCLAW_STATE_DIR;
  if (!stateDir) {
    console.log('[gateway] OPENCLAW_STATE_DIR not set, skipping gateway start');
    return;
  }

  const entry = process.env.OPENCLAW_ENTRY || '/openclaw/dist/entry.js';
  const token = process.env.OPENCLAW_GATEWAY_TOKEN || 'dev-token';

  console.log(`[gateway] Starting OpenClaw gateway on port ${INTERNAL_PORT}`);

  gatewayProcess = spawn('node', [
    entry, 'gateway', 'run',
    '--bind', 'loopback',
    '--port', String(INTERNAL_PORT),
    '--auth', 'token',
    '--token', token,
  ], {
    stdio: 'inherit',
    env: { ...process.env },
  });

  gatewayProcess.on('exit', (code) => {
    console.log(`[gateway] Process exited with code ${code}`);
    gatewayReady = false;
    // Auto-restart after 2 seconds
    setTimeout(startGateway, 2000);
  });

  // Probe for readiness
  const probe = setInterval(async () => {
    try {
      const res = await fetch(`http://127.0.0.1:${INTERNAL_PORT}/`);
      if (res.ok) {
        gatewayReady = true;
        clearInterval(probe);
        console.log('[gateway] Ready');
      }
    } catch {
      // Not ready yet
    }
  }, 2000);
}

// Simple HTTP server that proxies to gateway or serves health
const server = http.createServer((req, res) => {
  if (req.url === '/healthz') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', gateway: gatewayReady }));
    return;
  }

  if (!gatewayReady) {
    res.writeHead(503);
    res.end('Gateway not ready');
    return;
  }

  // Proxy to gateway
  const options = {
    hostname: '127.0.0.1',
    port: INTERNAL_PORT,
    path: req.url,
    method: req.method,
    headers: req.headers,
  };

  const proxy = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxy.on('error', () => {
    res.writeHead(502);
    res.end('Gateway unavailable');
  });

  req.pipe(proxy);
});

server.listen(PORT, () => {
  console.log(`[wrapper] Listening on port ${PORT}`);
  startGateway();
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('[wrapper] SIGTERM received, shutting down');
  if (gatewayProcess) gatewayProcess.kill('SIGTERM');
  server.close(() => process.exit(0));
});
