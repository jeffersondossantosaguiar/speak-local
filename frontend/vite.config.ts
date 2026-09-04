import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend runs on 127.0.0.1:8787 by default. Proxy /jobs and /health so the
// frontend can call relative URLs without CORS handling. /streams needs both
// REST and WebSocket proxying (ws: true upgrades the /streams/{id}/ws socket).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/jobs": "http://127.0.0.1:8787",
      "/health": "http://127.0.0.1:8787",
      "/streams": {
        target: "http://127.0.0.1:8787",
        ws: true,
      },
    },
  },
});
