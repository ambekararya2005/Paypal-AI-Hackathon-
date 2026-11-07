"""Run L0 over the dataset, and verify the Day 3 bucket contract against it.

    python -m mantis.defense.l0_rules

Two outputs:

1. **Per-clause precision and false-positive rate on legitimate traffic.** A
   protocol rule with a bad FP rate gets switched off in week one, so the FP rate
   is the number that decides whether a clause is deployable, and it is reported
   per clause rather than only in aggregate.

2. **The bucket-contract verdict.** Day 3 asserted that CLEAN attacks trip zero
   L0 clauses and HARD attacks fire at least one on >=25% of events — but it
   asserted it against a *provisional* L0 that did not exist yet. This is the
   first time that claim meets a real implementation, and it is the only honest
   moment to check it. If the two disagree, the disagreement is reported rather
   than adjusted away in either direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mantis.core.paths import GENERATED_DIR
from mantis.defense.l0_rules.rules import (
    CLAUSES,
    evaluate,
    make_untrusted_domain_clause,
    trusted_domains,
)

#: Injectors declaring themselves CLEAN in ``foundry/injectors/agentic.py``.
CLEAN_ATTACKS: tuple[str, ...] = ("F1-01", "F1-03")

#: Injectors declaring themselves HARD. Each must fire a clause on at least
#: ``HARD_MIN_RATE`` of its events, per the Day 3 contract.
HARD_ATTACKS: tuple[str, ...] = ("F1-02", "F1-04", "F1-05", "F1-09", "F1-10")

#: The Day 3 threshold, restated here rather than imported so this file can be
#: read on its own.
HARD_MIN_RATE: float = 0.25

#: The one card known not to satisfy the Day 3 contract against a deployable L0.
#:
#: Its failure is a **finding**, reconciled in full below and pinned by
#: ``tests/test_l0_rules.py``, not a regression -- so it must not fail the build.
#: Any *other* card failing is a regression and does exit non-zero. That
#: distinction is the whole reason this is a named constant rather than a
#: blanket ``return 0``: if F1-05 is ever fixed, or if a second card starts
#: failing, the exit code changes and somebody has to look.
KNOWN_CONTRACT_EXCEPTION: str = "F1-05"


def clause_table(frame: pd.DataFrame, result) -> str:
    """Per-clause fire rate, precision and FP rate on legitimate traffic."""
    y = frame["is_fraud"].to_numpy()
    agentic = frame["ag_agent_id"].notna().to_numpy()
    n_legit_agentic = int((~y & agentic).sum())

    lines = [
        f"  {'clause':<28} {'fires':>7} {'precision':>10} {'FP rate':>9} "
        f"{'FP/legit agentic':>17}  status",
        f"  {'-' * 28} {'-' * 7} {'-' * 10} {'-' * 9} {'-' * 17}  {'-' * 8}",
    ]
    for name, mask in result.masks.items():
        fires = int(mask.sum())
        tp = int((mask & y).sum())
        fp = int((mask & ~y).sum())
        precision = tp / fires if fires else float("nan")
        fp_rate = fp / max(int((~y).sum()), 1)
        fp_agentic = fp / max(n_legit_agentic, 1)
        declared = name not in {c.name for c in CLAUSES}
        status = "DECLARED" if declared else "operative"
        lines.append(
            f"  {name:<28} {fires:>7,} {precision:>10.3f} {fp_rate:>9.5f} "
            f"{fp_agentic:>17.5f}  {status}"
        )

    fires = int(result.fired.sum())
    tp = int((result.fired & y).sum())
    fp = int((result.fired & ~y).sum())
    lines += [
        f"  {'-' * 28} {'-' * 7} {'-' * 10} {'-' * 9} {'-' * 17}  {'-' * 8}",
        f"  {'ANY operative clause':<28} {fires:>7,} {tp / max(fires, 1):>10.3f} "
        f"{fp / max(int((~y).sum()), 1):>9.5f} {fp / max(n_legit_agentic, 1):>17.5f}  fired",
    ]
    return "\n".join(lines)


def per_attack_table(frame: pd.DataFrame, result) -> str:
    """Which clause each attack trips, and on what share of its events."""
    y = frame["is_fraud"].to_numpy()
    attack_ids = frame["attack_id"].fillna("").to_numpy()
    names = [c.name for c in CLAUSES]

    lines = [
        f"  {'attack':<8} {'bucket':<7} {'n':>6} {'L0 recall':>10}  strongest clause",
        f"  {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 10}  {'-' * 46}",
    ]
    for attack in sorted(set(attack_ids[y])):
        rows = attack_ids == attack
        n = int(rows.sum())
        bucket = (
            "CLEAN"
            if attack in CLEAN_ATTACKS
            else "HARD"
            if attack in HARD_ATTACKS
            else "-"
        )
        recall = float(result.fired[rows].mean())
        best = sorted(
            ((float(result.masks[name][rows].mean()), name) for name in names), reverse=True
        )[0]
        detail = f"{best[1]} {best[0]:.2f}" if best[0] > 0 else "(none fire)"
        lines.append(f"  {attack:<8} {bucket:<7} {n:>6,} {recall:>10.3f}  {detail}")
    return "\n".join(lines)


def bucket_verdict(frame: pd.DataFrame, result) -> tuple[str, bool, list[str]]:
    """Check the Day 3 HARD/CLEAN contract against this L0.

    Returns ``(text, ok, failing_cards)``. The caller decides what an expected
    failure means for the exit code; see :data:`KNOWN_CONTRACT_EXCEPTION`.
    """
    attack_ids = frame["attack_id"].fillna("").to_numpy()
    names = [c.name for c in CLAUSES]
    lines: list[str] = []
    ok = True

    failed: list[str] = []
    lines.append("  CLEAN attacks must trip ZERO operative clauses, at zero tolerance.")
    lines.append("")
    for attack in CLEAN_ATTACKS:
        rows = attack_ids == attack
        if not rows.any():
            continue
        rate = float(result.fired[rows].mean())
        verdict = "PASS" if rate == 0.0 else "FAIL"
        ok &= rate == 0.0
        if rate > 0.0:
            failed.append(attack)
        lines.append(
            f"    {attack}  fired on {rate:>6.2%} of {int(rows.sum()):,} events   {verdict}"
        )
        if rate > 0:
            for name in names:
                share = float(result.masks[name][rows].mean())
                if share > 0:
                    lines.append(f"        via {name}: {share:.2%}")

    lines += [
        "",
        f"  HARD attacks must fire a clause on >= {HARD_MIN_RATE:.0%} of their events.",
        "",
    ]
    for attack in HARD_ATTACKS:
        rows = attack_ids == attack
        if not rows.any():
            continue
        best_share, best_name = sorted(
            ((float(result.masks[name][rows].mean()), name) for name in names), reverse=True
        )[0]
        verdict = "PASS" if best_share >= HARD_MIN_RATE else "FAIL"
        ok &= best_share >= HARD_MIN_RATE
        if verdict == "FAIL":
            failed.append(attack)
        lines.append(
            f"    {attack}  best clause {best_name:<20} {best_share:>6.2%}   {verdict}"
        )

    if failed:
        lines += ["", "  " + "=" * 76, "  WHICH SIDE IS WRONG", "  " + "=" * 76, ""]
        lines += _delegation_tradeoff(frame, failed)
    return "\n".join(lines), ok, failed


def _delegation_tradeoff(frame: pd.DataFrame, failed: list[str]) -> list[str]:
    """Price the Day 3 threshold in false positives, which is what settles it.

    The Day 3 test let a HARD card satisfy the contract with **any** signal that
    reached 25% recall, and never asked what that signal cost. For F1-05 the only
    signal clearing the bar is ``delegation_depth > 2``, and this is the table
    showing what it charges.
    """
    ids = frame["attack_id"].fillna("").to_numpy()
    agentic = frame["ag_agent_id"].notna().to_numpy()
    y = frame["is_fraud"].to_numpy()
    legit_agentic = agentic & ~y
    depth = frame["ag_delegation_depth"].to_numpy(dtype=float)

    lines = [
        "  The Day 3 test satisfied F1-05 with `delegation_depth > 2`. That threshold",
        "  was never priced. Pricing it:",
        "",
        f"    {'threshold':<14} {'F1-05 recall':>13} {'FP on legit agentic':>21} {'FP count':>10}",
        f"    {'-' * 14} {'-' * 13} {'-' * 21} {'-' * 10}",
    ]
    target = ids == "F1-05"
    for t in (2, 3, 4, 5):
        with np.errstate(invalid="ignore"):
            fires = depth > t
        recall = float(fires[target].mean()) if target.any() else float("nan")
        fp_rate = float(fires[legit_agentic].mean())
        lines.append(
            f"    depth > {t:<7}{recall:>14.3f} {fp_rate:>21.5f} "
            f"{int((fires & legit_agentic).sum()):>10,}"
        )

    lines += [
        "",
        "  VERDICT: the Day 3 CONTRACT is wrong, not L0.",
        "",
        "  L0's defining property is that it fires on provable protocol violations at",
        "  near-zero false-positive rate. That is what makes it deployable tomorrow and",
        "  what makes its errors policy arguments rather than mistakes. `depth > 2`",
        "  declines 5% of all legitimate agent-mediated authorisations. No issuer ships",
        "  that, so it is not an L0 clause -- it is a weak classifier with a rule's",
        "  syntax. At the only threshold that costs nothing (`depth > 5`, just above the",
        "  legitimate tail the reference was widened to on Day 3), F1-05 fires on 3%.",
        "",
        "  The Day 3 test had no false-positive term at all: it accepted any signal",
        "  reaching 25% recall. Four of the five HARD cards survive that omission",
        "  because their clauses genuinely are free -- an expired mandate is expired, an",
        "  invalid signature is invalid, and neither fires on legitimate traffic at all.",
        "  F1-05 is the one card where the omission was load-bearing.",
        "",
        "  And F1-05's own docstring already says so: 'depth alone will not carry this",
        "  card', 'it is the weakest of the HARD signals', and the real answer named",
        "  there is the L4 graph layer -- sub-agent fan-out, mandate-hash reuse across",
        "  unrelated agents. The declaration was aspirational about a layer that does",
        "  not exist yet. F1-05 behaves like a third category the Day 3 binary had no",
        "  room for: an attack whose deterministic signal exists but is not separable",
        "  at any usable operating point.",
        "",
        "  NEITHER SIDE WAS ADJUSTED TO AGREE. F1-05 still declares HARD, L0 still",
        "  thresholds at 5, and this block is the reconciliation. Fixing it means one",
        "  of two things: add an FP term to the Day 3 test and reclassify F1-05, or",
        "  build L4 and give the card the signal it was always relying on.",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mantis.defense.l0_rules")
    parser.add_argument("--dataset", type=Path, default=GENERATED_DIR / "dataset_v1.parquet")
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"no dataset at {args.dataset}; run `make dataset` first", file=sys.stderr)
        return 1

    frame = pd.read_parquet(args.dataset).sort_values("ts").reset_index(drop=True)
    allow = trusted_domains(frame)
    clauses = (*CLAUSES, make_untrusted_domain_clause(allow))
    result = evaluate(frame, clauses)

    print("=" * 84)
    print("L0 - deterministic protocol-integrity clauses")
    print("=" * 84)
    agentic = int(frame["ag_agent_id"].notna().sum())
    print(f"  {len(frame):,} events, {int(frame['is_fraud'].sum()):,} fraud, "
          f"{agentic:,} carrying an agentic block")
    print()
    print(clause_table(frame, result))
    print()
    print("  'FP rate' is over ALL legitimate traffic; 'FP/legit agentic' over the rows")
    print("  a clause can actually fire on. The second is the honest denominator for a")
    print("  rule that only reads mandates, and it is the one an issuer would quote.")
    print()

    print("-" * 84)
    print("per attack")
    print("-" * 84)
    print(per_attack_table(frame, result))
    print()

    print("=" * 84)
    print("THE DAY 3 BUCKET CONTRACT, CHECKED AGAINST A REAL L0 FOR THE FIRST TIME")
    print("=" * 84)
    text, ok, failures = bucket_verdict(frame, result)
    print(text)
    print()
    print(f"  VERDICT: {'the contract holds' if ok else 'THE CONTRACT IS VIOLATED'}")
    unexpected = sorted(set(failures) - {KNOWN_CONTRACT_EXCEPTION})
    if failures and not unexpected:
        print(f"           (the failure is {KNOWN_CONTRACT_EXCEPTION}, which is the documented")
        print("            Day 4 finding above and is pinned by tests/test_l0_rules.py --")
        print("            so this run exits 0. Any other card failing exits non-zero.)")
    elif unexpected:
        print(f"           UNEXPECTED failures: {unexpected}")
    print()

    declared = result.masks.get("provenance_untrusted_domain")
    if declared is not None:
        y = frame["is_fraud"].to_numpy()
        ids = frame["attack_id"].fillna("").to_numpy()
        print("-" * 84)
        print("the clause that is declared and switched OFF")
        print("-" * 84)
        fp = int((declared & ~y).sum())
        print(f"  provenance_untrusted_domain would fire on {int(declared.sum()):,} events, "
              f"{fp:,} of them legitimate.")
        for attack in (*CLEAN_ATTACKS, *HARD_ATTACKS):
            rows = ids == attack
            if rows.any():
                print(f"    {attack}: {float(declared[rows].mean()):.0%}")
        print()
        print("  It is excluded because that number is a fact about the generator, not")
        print("  about the attack: the foundry draws attacker URLs from twelve hosts that")
        print("  appear nowhere in legitimate traffic. A real stream does not hand you")
        print("  that partition, and trusting it here would let L3 post a recall it had")
        print("  not earned by reading a single word. See l0_rules/rules.py.")
    return 0 if not unexpected else 2


if __name__ == "__main__":
    sys.exit(main())
