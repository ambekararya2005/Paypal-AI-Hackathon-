import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// Self-hosted fonts (OFL, @fontsource) at fixed /fonts/ URLs, Latin subset, only
// the weights in use. Fixed rather than hashed names so the server-rendered
// consent page (mantis/api/consent.py) can load the very same files from the
// same origin. Dev serves them from node_modules; build emits them into dist.
const FONTS: Record<string, string> = Object.fromEntries(
  [
    ["ibm-plex-sans", "ibm-plex-sans-latin-400-normal"],
    ["ibm-plex-sans", "ibm-plex-sans-latin-500-normal"],
    ["ibm-plex-sans", "ibm-plex-sans-latin-600-normal"],
    ["ibm-plex-mono", "ibm-plex-mono-latin-400-normal"],
    ["ibm-plex-mono", "ibm-plex-mono-latin-500-normal"],
  ].map(([pkg, name]) => [
    `${name}.woff2`,
    fileURLToPath(new URL(`./node_modules/@fontsource/${pkg}/files/${name}.woff2`, import.meta.url)),
  ]),
);

function selfHostedFonts(): Plugin {
  return {
    name: "mantis-fonts",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const m = /^\/fonts\/([\w-]+\.woff2)$/.exec(req.url ?? "");
        const file = m && FONTS[m[1]];
        if (!file) return next();
        res.setHeader("Content-Type", "font/woff2");
        res.end(readFileSync(file));
      });
    },
    generateBundle() {
      for (const [name, file] of Object.entries(FONTS)) {
        this.emitFile({ type: "asset", fileName: `fonts/${name}`, source: readFileSync(file) });
      }
    },
  };
}

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
  plugins: [react(), selfHostedFonts()],
  // Read VITE_* from the repo-root .env (only VITE_-prefixed names reach the
  // bundle). Locally that is VITE_AG_STUDIO_LICENSE_KEY; the Docker build has no
  // .env, and the deployed console fetches the key from /api/client-config.
  envDir: "..",
  // The same proxy on both servers. Without it on `preview`, a judge who runs
  // the production build locally gets a page that loads and then fails every
  // fetch, which looks far more broken than it is.
  server: { port: 5173, proxy: PROXY },
  preview: { port: 4173, proxy: PROXY },
  build: { outDir: "dist", sourcemap: false },
});
