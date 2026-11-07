import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API base is baked at build time from VITE_API_BASE so that the same
// source deploys to Vercel against a hosted API and runs locally against
// 127.0.0.1:8000 with no code change. The dev proxy exists so a developer does
// not have to think about CORS at all while iterating.
const PROXY = {
  "/api": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  // The same proxy on both servers. Without it on `preview`, a judge who runs
  // the production build locally gets a page that loads and then fails every
  // fetch, which looks far more broken than it is.
  server: { port: 5173, proxy: PROXY },
  preview: { port: 4173, proxy: PROXY },
  build: { outDir: "dist", sourcemap: false },
});
