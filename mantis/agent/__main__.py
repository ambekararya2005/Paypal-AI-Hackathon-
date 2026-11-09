"""``python -m mantis.agent`` — the LLM shopping agent against the gate.

    python -m mantis.agent demo                         # mock: replays every turn, no keys
    python -m mantis.agent demo --mode live             # live LLM + PayPal sandbox; records
    python -m mantis.agent demo --scenarios D E         # a subset
    python -m mantis.agent demo --agent-profile naive   # SIMULATED compromised agent
    python -m mantis.agent demo --serve                 # leave G held; open the consent page

``--llm`` and ``--paypal`` override the two halves separately (``--llm live
--paypal mock`` records agent and judge turns without touching the sandbox — the
tool results the model sees are identical either way).

Exit status is non-zero when any scenario's check fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from mantis.agent.scenarios import AGENT_SCENARIOS, TASK

_RULE = "-" * 120


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def build_gate(paypal_mode: str, llm_mode: str) -> Any:  # Any: PaymentGate, imported lazily
    from mantis.defense.l3_judge import InjectionJudge
    from mantis.defense.live import LiveScorer
    from mantis.paypal.config import load_config
    from mantis.paypal.gate import PaymentGate

    # L1 pinned off: its reason text would reach the model's next turn only through
    # the decision, but its presence is a local, gitignored artefact. Pinning it
    # makes a recorded run replay identically on a clean clone.
    scorer = LiveScorer(use_l1=False, judge=InjectionJudge(llm_mode))  # type: ignore[arg-type]
    return PaymentGate(load_config(paypal_mode), scorer=scorer)


def run_demo(args: argparse.Namespace) -> int:
    from mantis.agent.shopper import Shopper, ToolLayer, agent_params
    from mantis.agent.store import Store
    from mantis.llm.client import CacheMissError, LlmChat
    from mantis.paypal.scenarios import AGENT_ID, AGENT_PLATFORM, KYA_TOKEN

    llm_mode = args.llm or args.mode
    paypal_mode = args.paypal or args.mode
    started = time.perf_counter()
    gate = build_gate(paypal_mode, llm_mode)
    chat_probe = LlmChat("agent", mode=llm_mode)
    model = args.agent_model or chat_probe.model
    print(f"MANTIS agent demo  [LLM={llm_mode}, PayPal={paypal_mode}]  "
          f"agent={model}  judge={gate.scorer.judge.model}  "
          f"profile={args.agent_profile}"
          + ("  ** SIMULATED COMPROMISED AGENT **" if args.agent_profile == "naive" else ""))
    print(f"  gate loaded in {time.perf_counter() - started:.1f}s; L3 abstains below vocabulary "
          f"coverage {gate.scorer.l3_min_coverage:.2f}; L1 pinned off for replay")
    from mantis.agent.mandate_compiler import MandateCompiler
    from mantis.agent.scenarios import MANDATE_TEXT

    compiled = MandateCompiler(llm_mode).compile(MANDATE_TEXT)  # type: ignore[arg-type]
    print(f'  mandate, compiled from the sentence "{MANDATE_TEXT}"  [{compiled.source}]')
    for line in compiled.summary().splitlines():
        print(f"    {line}")
    if compiled.status != "ok":
        return 2
    print(f"  task: {TASK}")
    print(_RULE)
    print(f"{'':1}{'sc':<3}{'agent final action':<34}{'decision':<9}{'layers_ran':<22}"
          f"{'PayPal':<9}{'order id':<19}{'wall s':>7}{'judge s':>8}")
    print(_RULE)

    failures, held = 0, []
    wanted = {s.upper() for s in args.scenarios} if args.scenarios else None
    for scenario in AGENT_SCENARIOS:
        if wanted and scenario.key not in wanted:
            continue
        mandate = gate.create_mandate(
            compiled.request.model_copy(update={"vault_id": gate.vault_id()}))
        settings = None
        chat = LlmChat("agent", mode=llm_mode, refresh=args.refresh, settings=settings)
        if args.agent_model:
            chat.model = args.agent_model
        tools = ToolLayer(store=Store(scenario.variant),  # type: ignore[arg-type]
                          gate=gate, mandate_id=mandate.mandate_id, agent_id=AGENT_ID,
                          agent_platform=AGENT_PLATFORM, kya_token=KYA_TOKEN)
        shopper = Shopper(chat, tools, profile=args.agent_profile,
                          params=agent_params(chat.model))
        t0 = time.perf_counter()
        try:
            run = shopper.run(TASK)
        except CacheMissError as exc:
            print(f"!{scenario.key:<3}not recorded: {exc}")
            failures += 1
            continue
        wall = time.perf_counter() - t0
        judge_s = sum(r.judge_ms for r in run.checkouts) / 1000.0
        ok, note = scenario.check(run)
        failures += not ok

        rows = run.checkouts or [None]
        for i, record in enumerate(rows):
            label = f"{scenario.key}" if i == 0 else ""
            if record is None:
                print(f"{'' if ok else '!'}{label:<3} no checkout: {_short(run.final_text, 60)}")
                continue
            action = f"pay {record.merchant_id[2:]} ${record.amount}"
            print(f"{'' if ok else '!'}{label:<3}{action:<34}{record.decision.upper():<9}"
                  f"{'+'.join(record.layers_ran):<22}{record.paypal_action:<9}"
                  f"{record.order_id:<19}"
                  + (f"{wall:>7.1f}{judge_s:>8.1f}" if i == 0 else ""))
            for reason in record.reasons:
                print(f"{'':<5}[{reason['layer']}] {reason['code']}: "
                      f"{_short(reason['detail'], 100)}")
            if record.status == "held":
                held.append(record)
                if scenario.consent is not None and not args.serve:
                    gate.consent(record.order_id, approve=scenario.consent, signer="demo-human")
                    print(f"{'':<5}consent: demo human "
                          f"{'APPROVED -> captured' if scenario.consent else 'DENIED -> voided'}")
        print(f"{'':<5}{scenario.title}: {'PASS' if ok else 'FAIL'} - {note}")
        print(f"{'':<5}agent: {run.turns} turns, {len(run.pages)} pages read, "
              f"{len(run.tool_log)} tool calls, LLM {run.llm_ms / 1000:.1f}s "
              f"({run.cache_hits} cached / {run.live_calls} live), stopped={run.stopped}")
        print(_RULE)

    total = len(wanted) if wanted else len(AGENT_SCENARIOS)
    print(f"{total - failures}/{total} scenarios behaved as declared. wall s = whole scenario "
          "(agent + gate + PayPal); judge s = the LLM judge, reported apart from scoring.")
    if args.serve and held:
        return _serve(gate, held)
    return 1 if failures else 0


def _serve(gate: Any, held: list[Any]) -> int:  # Any: PaymentGate / GateRecord
    import uvicorn

    import mantis.api.gate as api_gate
    from mantis.api.app import app

    api_gate._GATE = gate
    print("\nHeld orders are waiting for a human. Open:")
    for record in held:
        print(f"  http://127.0.0.1:8000/gate/{record.order_id}/consent")
    print("Ctrl+C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run scenarios D/E/F/G")
    demo.add_argument("--mode", choices=("mock", "live"), default="mock")
    demo.add_argument("--llm", choices=("mock", "live"), default=None)
    demo.add_argument("--paypal", choices=("mock", "live"), default=None)
    demo.add_argument("--scenarios", nargs="*", default=None)
    demo.add_argument("--agent-profile", choices=("standard", "naive"), default="standard")
    demo.add_argument("--agent-model", default=None, help="override LLM_MODEL_AGENT")
    demo.add_argument("--refresh", action="store_true", help="live: ignore cached turns")
    demo.add_argument("--serve", action="store_true",
                      help="leave held orders pending and serve the consent page on :8000")
    args = parser.parse_args(argv)
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
