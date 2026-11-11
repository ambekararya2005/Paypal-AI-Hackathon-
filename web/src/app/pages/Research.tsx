/** Offline research: the four pre-hackathon screens, unchanged, as sub-tabs. */

import Atlas from "../../Atlas";
import Console from "../../Console";
import Fidelity from "../../Fidelity";
import Results from "../../Results";
import type { ResearchTab } from "../router";
import { PageHead } from "../ui";

const TABS: [ResearchTab, string, () => JSX.Element][] = [
  ["console", "Live console", Console],
  ["results", "Results", Results],
  ["atlas", "Attack atlas", Atlas],
  ["fidelity", "Fidelity", Fidelity],
];

export default function Research({ tab }: { tab: ResearchTab }) {
  const Screen = TABS.find(([id]) => id === tab)![2];
  return (
    <>
      <PageHead title="Offline research" />
      <div className="m-banner-muted">Offline research on synthetic data (pre-hackathon)</div>
      <nav className="m-tabs" aria-label="Research">
        {TABS.map(([id, label]) => (
          <a key={id} href={`#/research/${id}`} aria-current={id === tab ? "page" : undefined}>
            {label}
          </a>
        ))}
      </nav>
      <div className="research">
        <Screen />
      </div>
    </>
  );
}
