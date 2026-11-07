"""Build, inspect and sample the content corpus.

    python -m mantis.foundry.llm                 # build (cache/fallback), summarise
    python -m mantis.foundry.llm --show          # print one specimen of each kind
    python -m mantis.foundry.llm --live --refresh   # author it against Ollama

The default invocation touches no network. ``--live`` permits one, ``--refresh``
forces regeneration rather than reading the cache; both are development flags,
and the committed corpus is what a demo actually reads.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from mantis.foundry.llm.build import CORPUS_PLAN, build_corpus
from mantis.foundry.llm.client import DEFAULT_MODEL, LIVE_ENV_VAR, LlmClient
from mantis.foundry.llm.corpus import ContentStore, content_id_for_url


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mantis.foundry.llm",
        description="Build the committed content corpus for the L3 text layer.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    parser.add_argument(
        "--live",
        action="store_true",
        help=f"permit live generation (equivalent to {LIVE_ENV_VAR}=1)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached generations and regenerate"
    )
    parser.add_argument("--show", action="store_true", help="print one specimen per kind")
    parser.add_argument("--no-write", action="store_true", help="do not write the corpus files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and report the corpus."""
    args = _parse_args(argv)
    client = LlmClient(model=args.model, allow_live=True if args.live else None)

    print("MANTIS content corpus build")
    print(f"  model          : {client.model}")
    print(f"  live permitted : {client.live_enabled}")
    print(f"  live reachable : {client.live_reachable()}")
    print(f"  cache          : {len(client.cache)} entries -> {client.cache.directory}")
    print()

    store = build_corpus(client, ContentStore(), refresh=args.refresh)

    counts = store.counts()
    print("artefacts by kind")
    for spec in CORPUS_PLAN:
        print(f"  {spec.kind:<22} {counts.get(spec.kind, 0):>4}")
    print(f"  {'-' * 22} {'-' * 4}")
    print(f"  {'benign':<22} {counts['_benign']:>4}")
    print(f"  {'adversarial':<22} {counts['_injected']:>4}")
    print(f"  {'total':<22} {len(store.artifacts):>4}")
    print()
    print(f"generation sources: {client.summary()}")
    print()

    if args.show:
        print("=" * 78)
        for spec in CORPUS_PLAN:
            artifact = store.pick(spec.kind, 0)
            flag = "ADVERSARIAL" if artifact.is_injected else "benign"
            print(f"[{flag}] {artifact.artifact_id}  ({artifact.title})  src={artifact.source}")
            for line in artifact.text.splitlines():
                print(textwrap.fill(line, 74, initial_indent="    ", subsequent_indent="    "))
            print()
        print("=" * 78)
        print()

    if not args.no_write:
        path = store.write()
        size_kb = path.stat().st_size / 1024
        print(f"wrote {len(store.artifacts)} artefacts -> {path}  ({size_kb:.1f} KB)")
        print(f"      bindings   -> {path.with_name('bindings.jsonl')}")

    # The join, demonstrated rather than asserted: an id computed the way the
    # simulator computes it must resolve to text.
    demo_id = content_id_for_url("https://shop.example.test/product/abc123")
    resolved = store.resolve(demo_id)
    print()
    print("join check (this is what L3 will do on every event):")
    print("  url        https://shop.example.test/product/abc123")
    print(f"  content_id {demo_id}")
    print(f"  resolves   {resolved.artifact_id if resolved else 'MISS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
