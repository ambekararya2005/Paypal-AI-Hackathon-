import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
// Order matters: the research screens' stylesheet first, then the tokens (which
// win on any shared custom property), then the app.
import "./styles.css";
import "./tokens.css";
import "./app/app.css";
import "./ops/ops.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
