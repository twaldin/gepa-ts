import * as fs from 'node:fs';
import { create_server } from './server.js';

const socket_path = process.env['GEPA_TS_SIDECAR_SOCKET'] ?? '/tmp/gepa-ts.sock';

try {
  fs.unlinkSync(socket_path);
} catch {
  // stale socket may not exist
}

const server = create_server();

server.listen(socket_path, () => {
  process.stderr.write(`[gepa-ts-sidecar] listening on ${socket_path}\n`);
});

function shutdown(): void {
  server.close(() => {
    try {
      fs.unlinkSync(socket_path);
    } catch {
      // already gone
    }
    process.exit(0);
  });
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
