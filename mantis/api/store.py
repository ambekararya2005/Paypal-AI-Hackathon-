"""Everything the API serves, loaded once at import and never recomputed.

The brief for this API is one sentence: *serve arena.json and the RESULTS.md
numbers straight from disk — no recomputation at request time.* This module is
where that is enforced, so that no route handler is ever in a position to fit a
model.

Two consequences worth stating, because they are design decisions rather than
shortcuts:

**Startup is the only place that can be slow, and it is not slow.** Four JSON
reads and a markdown parse — well under a second. A judge's first request is not
paying for a cold model.

**Every artefact is optional and reports its own absence.** A clean clone that
has never run ``make firewall`` still starts, still serves the atlas (which is
committed source, not generated), and answers ``/results`` with a 503 naming the
command to run. The alternative — an import-time crash — turns a missing optional
artefact into a dead API, and the one thing worse than a console with a blank
panel is a console that will not load at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from mantis.atlas.loader import ATLAS, DISCOVERED
from mantis.core.events import SCHEMA_VERSION
from mantis.core.paths import DOCS_DIR, GENERATED_DIR, REPO_ROOT
from mantis.defense import results_doc

__all__ = ["STORE", "Store"]

ARENA_JSON: Final[Path] = GENERATED_DIR / "arena.json"
FEED_JSON: Final[Path] = GENERATED_DIR / "console_feed.json"
OOD_JSON: Final[Path] = GENERATED_DIR / "l3_ood.json"
FIDELITY_JSON: Final[Path] = GENERATED_DIR / "fidelity.json"
LATENCY_JSON: Final[Path] = GENERATED_DIR / "latency.json"
POPULATION_MANIFEST: Final[Path] = GENERATED_DIR / "population.manifest.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON artefact, or ``None`` if it is absent or unreadable.

    Unreadable is folded into absent on purpose. A half-written arena.json from
    an interrupted run should present as "the loop has not been run", which is
    recoverable and clearly signposted, rather than as a 500 from a route.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _float(text: str) -> float | None:
    """A number out of a RESULTS.md cell: ``0.450``, ``**0.539**``, ``66%``."""
    cleaned = text.replace("*", "").replace("%", "").replace(",", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100 if "%" in text else value


@dataclass(slots=True)
class Store:
    """The loaded artefacts, plus the small derived views the routes need."""

    results: results_doc.ResultsDoc | None = None
    results_error: str = ""
    arena: dict[str, Any] | None = None
    feed: dict[str, Any] | None = None
    ood: dict[str, Any] | None = None
    fidelity: dict[str, Any] | None = None
    latency: dict[str, Any] | None = None
    population: dict[str, Any] | None = None
    headline: dict[str, Any] = field(default_factory=dict)

    @property
    def events(self) -> list[dict[str, Any]]:
        return (self.feed or {}).get("events", [])

    @property
    def feed_manifest(self) -> dict[str, Any]:
        return (self.feed or {}).get("manifest", {})

    def load(self) -> Store:
        try:
            self.results = results_doc.load()
        except (FileNotFoundError, ValueError) as exc:
            self.results, self.results_error = None, str(exc)

        self.arena = _read_json(ARENA_JSON)
        self.feed = _read_json(FEED_JSON)
        self.ood = _read_json(OOD_JSON)
        self.fidelity = _read_json(FIDELITY_JSON)
        self.latency = _read_json(LATENCY_JSON)
        self.population = _read_json(POPULATION_MANIFEST)
        self.headline = self._build_headline()
        return self

    # ------------------------------------------------------------------ views --
    def _build_headline(self) -> dict[str, Any]:
        """The handful of numbers a slide quotes, extracted once.

        Pulled from RESULTS.md and arena.json rather than hard-coded, so a
        retrain moves the big number on the console without anyone editing
        TypeScript. Every field is nullable: a missing artefact renders a dash,
        not a zero, because a zero here would be a claim.
        """
        out: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "atlas_cards": len(ATLAS),
            "atlas_implemented": sum(1 for c in ATLAS.values() if c.generator),
            "discovered_cards": len(DISCOVERED),
        }

        if self.results is not None:
            layer = self.results.table_for("Layer performance")
            if layer is not None:
                for row in layer.as_records():
                    name = row.get("layer", "")
                    if name.startswith("L1"):
                        out["l1_auc_pr"] = _float(row.get("AUC-PR", ""))
                        out["l1_recall"] = _float(row.get("recall@0.1%", ""))
                    elif name.startswith("fused"):
                        out["fused_recall"] = _float(row.get("recall@0.1%", ""))
                        out["fused_campaign_recall"] = _float(
                            (row.get("campaign recall", "") or "").split(" ")[0]
                        )

            dataset = self.results.table_for("evaluation dataset")
            if dataset is not None:
                for row in dataset.rows:
                    if len(row) >= 2 and row[0] == "events":
                        out["n_events"] = _float(row[1])
                    if len(row) >= 2 and row[0] == "fraud":
                        out["n_fraud"] = _float(row[1].split(" ")[0])

            zero = self.results.table_for("zero-day demonstration")
            if zero is not None and zero.rows:
                values = [_float(r[1]) for r in zero.rows if len(r) > 1]
                keys = ("trained_with_family", "family_held_out", "loop_augmented")
                out["zero_day"] = dict(zip(keys, values, strict=False))

        if self.arena:
            out["evasion_curve"] = self.arena.get("evasion_curve", [])
            if self.arena.get("zero_day"):
                out["zero_day_detail"] = self.arena["zero_day"]

        if self.ood:
            out["l3_ood"] = {
                "in_roc": self.ood.get("in_distribution", {}).get("roc"),
                "out_roc": self.ood.get("out_of_distribution", {}).get("roc"),
                "out_fp": self.ood.get("out_of_distribution", {}).get("fp_rate"),
            }
        return out

    def table_records(self, needle: str) -> list[dict[str, str]]:
        """One RESULTS.md table as records, or an empty list if it is absent."""
        if self.results is None:
            return []
        table = self.results.table_for(needle)
        return table.as_records() if table else []

    def evasion_curve(self) -> list[float]:
        if self.arena and self.arena.get("evasion_curve"):
            return [float(v) for v in self.arena["evasion_curve"]]
        return []

    def docs_figure(self, name: str) -> Path | None:
        """A committed figure from ``docs/``, path-checked.

        ``name`` reaches this from a URL, so it is resolved and then required to
        sit inside ``docs/``. Without that check a request for
        ``../../.ssh/id_rsa`` would be served happily.
        """
        candidate = (DOCS_DIR / name).resolve()
        if not candidate.is_file() or DOCS_DIR.resolve() not in candidate.parents:
            return None
        return candidate


#: Loaded once, at import. Route handlers read this and never rebuild it.
STORE: Final[Store] = Store().load()


def main() -> None:
    """What the API can serve right now. ``python -m mantis.api.store``."""
    print("MANTIS API store")
    print("=" * 70)
    print(f"  repo            {REPO_ROOT}")
    print(f"  schema          v{SCHEMA_VERSION}")
    print(f"  atlas           {len(ATLAS)} cards, "
          f"{STORE.headline.get('atlas_implemented')} implemented, "
          f"{len(DISCOVERED)} discovered")
    for label, ok, hint in (
        ("RESULTS.md", STORE.results is not None, "make firewall"),
        ("arena.json", STORE.arena is not None, "make loop"),
        ("console_feed.json", STORE.feed is not None, "make feed"),
        ("l3_ood.json", STORE.ood is not None, "make ood"),
        ("fidelity.json", STORE.fidelity is not None, "make fidelity"),
        ("latency.json", STORE.latency is not None, "make latency"),
    ):
        mark = "ok     " if ok else "MISSING"
        print(f"  {label:<20} {mark}  {'' if ok else '-> ' + hint}")
    if STORE.feed:
        print(f"  feed events     {len(STORE.events)}")
    print()
    print("  headline:")
    for key, value in STORE.headline.items():
        print(f"    {key:<24} {value}")


if __name__ == "__main__":
    main()
