/**
 * MANTIS: the app. A hash router with real paths inside one shell.
 *
 *   #/overview            Overview (default)
 *   #/orders              Orders: the AG Studio dashboard over the gate log
 *   #/orders/:id          one order, laid out like a payment page
 *   #/review              held orders waiting on a person
 *   #/playground          write an attack, watch a real agent run (?replay=giftcard)
 *   #/verify?id=…         recompute a receipt hash and compare it with PayPal's
 *   #/mcp                 Claude Desktop (MCP): MANTIS as an MCP server
 *   #/research/:tab       the pre-hackathon offline research screens
 *
 * Old links (#ops, #attack, #console, #results, #atlas, #fidelity, #home) are
 * redirected in router.ts. Not a router library: CLAUDE.md §5 says nothing
 * heavier than the chart library.
 */

import OpsDashboard from "./ops/OpsDashboard";
import Playground from "./playground/Playground";
import Shell, { type Crumb } from "./app/Shell";
import { shortId } from "./app/data";
import Mcp from "./app/pages/Mcp";
import OrderDetail from "./app/pages/OrderDetail";
import Overview from "./app/pages/Overview";
import Research from "./app/pages/Research";
import ReviewQueue from "./app/pages/ReviewQueue";
import VerifyPage from "./app/pages/VerifyPage";
import { useRoute, type Route } from "./app/router";
import { Empty } from "./app/ui";

const TAB_LABEL = { console: "Live console", results: "Results", atlas: "Attack atlas", fidelity: "Fidelity" };

function crumbsFor(route: Route): Crumb[] {
  switch (route.name) {
    case "overview":
      return [["Overview"]];
    case "orders":
      return [["Orders"]];
    case "order":
      return [["Orders", "#/orders"], [shortId(route.id)]];
    case "review":
      return [["Review queue"]];
    case "playground":
      return [["Playground"]];
    case "verify":
      return [["Verify receipt"]];
    case "mcp":
      return [["Claude Desktop (MCP)"]];
    case "research":
      return [["Offline research", "#/research/console"], [TAB_LABEL[route.tab]]];
    default:
      return [["Not found"]];
  }
}

function Page({ route }: { route: Route }) {
  switch (route.name) {
    case "overview":
      return <Overview />;
    case "orders":
      return <OpsDashboard />;
    case "order":
      return <OrderDetail key={route.id} id={route.id} />;
    case "review":
      return <ReviewQueue />;
    case "playground":
      return <Playground />;
    case "verify":
      return <VerifyPage id={route.id} />;
    case "mcp":
      return <Mcp />;
    case "research":
      return <Research tab={route.tab} />;
    default:
      return (
        <Empty title="Page not found">
          <p>
            Nothing lives at <span className="mono">#{route.path}</span>. <a href="#/overview">Go to Overview →</a>
          </p>
        </Empty>
      );
  }
}

export default function App() {
  const route = useRoute();
  return (
    <Shell route={route} crumbs={crumbsFor(route)}>
      <Page route={route} />
    </Shell>
  );
}
