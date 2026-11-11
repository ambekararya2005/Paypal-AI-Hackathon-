/** MCP Server: run MANTIS as an MCP proxy in front of PayPal's hosted tools. */

import { GITHUB } from "../data";
import { CopyButton, PageHead, Panel } from "../ui";

/** Verbatim from docs/claude_desktop_config.json. */
const CONFIG = `{
  "mcpServers": {
    "mantis-paypal-gate": {
      "command": "<ABSOLUTE PATH TO YOUR PYTHON - see docs/mcp.md>",
      "args": ["-m", "mantis.paypal.mcp_proxy"],
      "env": {
        "MANTIS_PAYPAL_MODE": "mock",
        "MANTIS_STORE_VARIANT": "poisoned",
        "MANTIS_CONSENT_PORT": "8765",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}`;

/** From the tools table in docs/mcp.md. */
const TOOLS: [string, string][] = [
  ["create_mandate", "The person's terms (categories, per-order ceiling, merchants, total budget), signed by the gate."],
  ["compile_mandate", "The same, from one plain-English sentence."],
  ["search_products", "The demo store's catalogue (FreshMart, GreenLeaf Market, GiftHub)."],
  ["read_page", "A demo store page. The proxy records every page it serves; that record is the order's provenance."],
  ["create_order", "MANTIS's own. The gate authorizes the PayPal order, scores it, then captures, voids or holds it and returns a consent_url."],
  ["paypal_get_order", "Read-only passthrough to PayPal's hosted MCP server."],
  ["paypal_list_transactions", "Read-only passthrough (needs Transaction Search on the sandbox app, else PayPal returns 403)."],
  ["paypal_list_products", "Read-only passthrough."],
];

export default function Mcp() {
  return (
    <>
      <PageHead
        title="MCP Server"
        sub="MANTIS runs as an MCP server over stdio. Claude Desktop, or any MCP client, gets PayPal's tools with the gate in front of them."
      />
      <div className="m-stack">
        <Panel title="Claude Desktop config" actions={<CopyButton value={CONFIG} label="Copy config" />}>
          <p className="m-muted">
            Add this to <span className="mono">claude_desktop_config.json</span>, set <span className="mono">command</span> to your Python, and restart
            Claude Desktop. Windows and macOS paths are in the guide.
          </p>
          <pre className="m-pre m-pre-code">{CONFIG}</pre>
        </Panel>

        <Panel title="Tools" actions={<span className="m-muted num">{TOOLS.length} tools</span>} flush>
          <div className="m-table-wrap">
            <table className="m-table">
              <thead>
                <tr>
                  <th>Tool</th>
                  <th>What it does</th>
                </tr>
              </thead>
              <tbody>
                {TOOLS.map(([name, what]) => (
                  <tr key={name}>
                    <td className="mono m-nowrap">{name}</td>
                    <td>{what}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="m-pad m-note">
            There is no capture, pay, refund or void tool. Only the gate can capture or void, and a held order waits for a person on the
            consent page.
          </p>
        </Panel>

        <p>
          <a href={`${GITHUB}/blob/main/docs/mcp.md`} target="_blank" rel="noreferrer">
            Read docs/mcp.md on GitHub →
          </a>
        </p>
      </div>
    </>
  );
}
