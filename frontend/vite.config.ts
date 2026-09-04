import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend runs on 127.0.0.1:8787 by default. Proxy /jobs and /health so the
// frontend can call relative URLs without CORS handling.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/jobs": "http://127.0.0.1:8787",
      "/health": "http://127.0.0.1:8787",
    },
  },
});
