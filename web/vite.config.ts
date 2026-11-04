import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served from https://<user>.github.io/neurotrace/ on GitHub Pages.
export default defineConfig({
  base: process.env.SITE_BASE ?? "/neurotrace/",
  plugins: [react()],
  build: { chunkSizeWarningLimit: 900 },
});
