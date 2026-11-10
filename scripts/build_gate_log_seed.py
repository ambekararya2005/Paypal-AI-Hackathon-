"""Build the gate log's seed rows from the recorded Day 2 sandbox runs.

    python scripts/build_gate_log_seed.py

Inputs (local, gitignored): the six gate records the Day 2 live run wrote to
``data/receipts/<hash>.gate.json`` — real PayPal sandbox order ids, real final
states — and their receipts. Outputs (committed):

* ``data/fixtures/gate_log_seed.jsonl`` — one dashboard row per order, so the
  ops dashboard is never empty on a clean clone or a fresh deploy;
* ``data/fixtures/receipts/<hash>.json`` — the receipts, so ``Verify`` works
  on the seeded rows without the original machine.

Two fields the Day 2 records predate are filled from a **cache replay** of the
same scenarios (no LLM call, no PayPal call): the agent's measured LLM time at
checkout, and the mandate's plain-language summary.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Final

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mantis.api.gate_log import SEED_PATH, row_from_record  # noqa: E402
from mantis.core.paths import DATA_DIR  # noqa: E402

RECEIPTS: Final[Path] = DATA_DIR / "receipts"
FIXTURE_RECEIPTS: Final[Path] = DATA_DIR / "fixtures" / "receipts"

#: Day 2 live run (CLAUDE.md §8, "PayPal Day 2 gate"): order -> (scenario, merchant).
LIVE: Final[dict[str, tuple[str, str]]] = {
    "9DB39582P4348282B": ("D", "m-freshmart"),
    "97W73088UA860300H": ("E", "m-freshmart"),
    "5YM287057T143621E": ("E", "m-gifthub"),
    "35166960YB9367310": ("F", "m-freshmart"),
    "9WJ34823YD332522T": ("F", "m-greenleaf"),
    "04B78221X7733610F": ("G", "m-freshmart"),
}


def _replay() -> dict[tuple[str, str], dict[str, Any]]:
    """(scenario, merchant) -> {agent_ms, mandate} from a mock cache replay."""
    import copy
    import tempfile

    from mantis.agent.scenarios import AGENT_SCENARIOS, TASK, day2_mandate
    from mantis.agent.shopper import Shopper, ToolLayer, agent_params
    from mantis.agent.store import Store
    from mantis.api.log_routes import mandate_summary
    from mantis.defense.l3_judge import InjectionJudge
    from mantis.defense.live import LiveScorer
    from mantis.llm.client import LlmChat
    from mantis.paypal.config import PayPalConfig
    from mantis.paypal.gate import PaymentGate
    from mantis.paypal.scenarios import AGENT_ID, AGENT_PLATFORM, KYA_TOKEN

    out: dict[tuple[str, str], dict[str, Any]] = {}
    scorer = LiveScorer(use_l1=False, judge=InjectionJudge("mock"))
    with tempfile.TemporaryDirectory() as tmp:
        config = PayPalConfig(mode="mock", client_id="", client_secret="",
                              receipt_dir=Path(tmp))
        for scenario in AGENT_SCENARIOS:
            gate = PaymentGate(config, scorer=copy.copy(scorer))
            mandate = gate.create_mandate(day2_mandate(vault_id=gate.vault_id()))
            chat = LlmChat("agent", mode="mock")
            tools = ToolLayer(Store(scenario.variant), gate, mandate.mandate_id,  # type: ignore[arg-type]
                              AGENT_ID, AGENT_PLATFORM, KYA_TOKEN)
            run = Shopper(chat, tools, params=agent_params(chat.model)).run(TASK)
            for record in run.checkouts:
                out[(scenario.key, record.merchant_id)] = {
                    "agent_ms": record.agent_ms,
                    "mandate": mandate_summary(gate, mandate.mandate_id),
                }
    return out


def main() -> int:
    records: dict[str, dict[str, Any]] = {}
    for path in RECEIPTS.glob("*.gate.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("order_id") in LIVE:
            records[record["order_id"]] = record
    missing = set(LIVE) - set(records)
    if missing:
        print(f"missing Day 2 records for {sorted(missing)}; the seed needs the machine "
              "that ran `python -m mantis.agent demo --mode live`", file=sys.stderr)
        return 2

    replay = _replay()
    FIXTURE_RECEIPTS.mkdir(parents=True, exist_ok=True)
    lines = []
    for order_id, (scenario, merchant) in LIVE.items():
        record = records[order_id]
        extra = replay.get((scenario, merchant), {})
        record["agent_ms"] = extra.get("agent_ms", 0.0)
        row = row_from_record(record, extra.get("mandate"), source="seed")
        row["scenario"] = scenario
        lines.append(json.dumps({"row": row}, ensure_ascii=False))
        receipt = RECEIPTS / f"{record['receipt_hash']}.json"
        if receipt.exists():
            shutil.copy(receipt, FIXTURE_RECEIPTS / receipt.name)
        print(f"  {scenario} {order_id} {merchant:<12} {row['decision']:<8} "
              f"{row['final_state']:<9} chips={row['layers_fired']} "
              f"agent {row['latency']['agent_ms'] / 1000:.1f}s")
    SEED_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} rows -> {SEED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
