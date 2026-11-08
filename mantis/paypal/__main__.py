"""``python -m mantis.paypal`` — the gate demo and the sandbox recorder.

    python -m mantis.paypal demo                      # mock, no keys, clean clone
    python -m mantis.paypal demo --mode live          # PayPal sandbox, keys from .env
    python -m mantis.paypal record                    # live + rewrite the mock fixtures
    python -m mantis.paypal config                    # resolved mode, never a secret

Exit status is non-zero when any scenario lands on a decision other than the one
it declares, so ``make gate-demo`` is a real check and not a printout.
"""

from __future__ import annotations

import argparse
import sys
import time

from mantis.paypal.config import ConfigError, load_config
from mantis.paypal.scenarios import SCENARIOS, demo_mandate, scenario_inputs

_RULE = "-" * 118


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "..."


def run_demo(mode: str | None, *, record: bool = False, source: str = "auto") -> int:
    from mantis.paypal.gate import GateError, PaymentGate

    try:
        config = load_config(mode)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    gate = PaymentGate(config, payment_source=source, record=record)
    load_s = time.perf_counter() - started
    vault_id = None if source == "card" else gate.vault_id()
    if source == "vault" and vault_id is None:
        print("no active PayPal vault: run 'python -m mantis.paypal vault-setup', approve the "
              "link with a sandbox personal account, then 'vault-finish'", file=sys.stderr)
        return 2
    mandate = gate.create_mandate(demo_mandate(vault_id=vault_id))

    print(f"MANTIS x PayPal authorization gate  [mode={gate.mode}"
          f"{', recording fixtures' if record else ''}]")
    print(f"  scorer loaded in {load_s:.1f}s  |  L3 hold line {gate.scorer.l3_hold_at:.3f}"
          f"  |  L1 {'loaded' if gate.scorer.l1 else 'not loaded (L0 + L3 only)'}")
    print(f"  mandate {mandate.mandate_id}: {mandate.description}")
    print(f"    scope MCC {mandate.scope.categories}, merchants {mandate.scope.allowed_merchants},"
          f" ceiling ${mandate.scope.max_amount:.2f}, hash {mandate.digest()[:16]}...")
    print(_RULE)
    print(f"{'':2}{'scenario':<44}{'decision':<10}{'PayPal action':<22}"
          f"{'order id':<19}{'score ms':>9}{'PayPal ms':>11}")
    print(_RULE)

    failures = 0
    for scenario in SCENARIOS:
        intent, context = scenario_inputs(scenario.key, mandate.mandate_id)
        try:
            record_ = gate.submit(intent, context)
        except GateError as exc:
            print(f"  {scenario.key} {scenario.title:<41} ERROR {exc}")
            failures += 1
            continue
        action = record_.paypal_action
        if record_.status == "held" and scenario.consent is not None:
            gate.consent(record_.order_id, approve=scenario.consent, signer="demo-human")
            action = f"hold -> {'capture' if scenario.consent else 'void'} (human)"
        ok = record_.decision == scenario.expect_decision
        failures += not ok
        print(f"{'' if ok else '!'}{'':<2}{scenario.key} {scenario.title:<41}"
              f"{record_.decision.upper():<10}{action:<22}{record_.order_id:<19}"
              f"{record_.latency_ms:>9.1f}{record_.paypal_ms:>11.0f}")
        for reason in record_.reasons:
            print(f"{'':<6}[{reason['layer']}] {reason['code']}: "
                  f"{_shorten(reason['detail'], 96)}")
        print(f"{'':<6}status={record_.status}  capture={record_.capture_id or '-'}  "
              f"auth={record_.authorization_id}  layers={'+'.join(record_.layers_ran)}  "
              f"source={record_.payment_source}  receipt={record_.receipt_hash[:16]}...")
        print(_RULE)

    print(f"{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios landed on their declared "
          f"decision.  score ms = MANTIS scoring only; "
          f"PayPal ms = PayPal round trips ({'sandbox' if gate.mode == 'live' else 'replay'}).")
    if gate.mode == "live":
        print("Live orders are in the sandbox business account: "
              "developer.paypal.com -> Sandbox accounts -> the business account -> Activity.")
    return 1 if failures else 0


def run_vault(command: str) -> int:
    """The one-time human approval, in two halves around a browser click."""
    from mantis.paypal.client import LiveClient, PayPalError
    from mantis.paypal.vault import VAULT_FILE, finish_vault, start_vault

    try:
        client = LiveClient(load_config("live"))
    except (ConfigError, ValueError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        if command == "vault-setup":
            state = start_vault(client, description="MANTIS agent mandate: weekly groceries")
            print("PayPal Vault setup token created.")
            print(f"  1. Open this link and log in with your SANDBOX PERSONAL (buyer) account:\n"
                  f"     {state['approve_url']}")
            print("  2. Approve. The page you land on afterwards does not matter.")
            print("  3. Run: python -m mantis.paypal vault-finish")
        else:
            state = finish_vault(client)
            print(f"Vault active. vault_id {state['vault_id'][:6]}... saved to {VAULT_FILE}")
            print("Agent orders now use payment_source.paypal.vault_id: "
                  "python -m mantis.paypal demo --mode live")
    except PayPalError as exc:
        print(f"PayPal refused: {exc}", file=sys.stderr)
        if command == "vault-finish":
            print("  (if the issue is about approval: open the link from vault-setup first)",
                  file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.paypal", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the three gate scenarios")
    demo.add_argument("--mode", choices=("mock", "live"), default=None,
                      help="override MANTIS_PAYPAL_MODE")
    demo.add_argument("--record", action="store_true",
                      help="live only: rewrite data/fixtures/paypal/ from sandbox responses")
    demo.add_argument("--source", choices=("auto", "card", "vault"), default="auto",
                      help="payer approval: Vault token or sandbox test card")
    sub.add_parser("record", help="shorthand for: demo --mode live --record")
    sub.add_parser("config", help="print the resolved configuration")
    sub.add_parser("vault-setup", help="live: start a PayPal Vault; prints the approval link")
    sub.add_parser("vault-finish", help="live: exchange the approved token for a vault_id")

    args = parser.parse_args(argv)
    if args.command == "config":
        from mantis.paypal.config import main as config_main

        config_main()
        return 0
    if args.command in ("vault-setup", "vault-finish"):
        return run_vault(args.command)
    if args.command == "record":
        return run_demo("live", record=True)
    if args.record and (args.mode or load_config().mode) != "live":
        print("--record needs --mode live", file=sys.stderr)
        return 2
    return run_demo(args.mode, record=args.record, source=args.source)


if __name__ == "__main__":
    sys.exit(main())
