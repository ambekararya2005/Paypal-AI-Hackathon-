"""Read ``RESULTS.md`` back into structure.

``report.py`` writes that document from a fitted experiment. This module reads it
back, and it exists so that **nothing downstream ever retypes a number**.

The rule it enforces
----------------------
Two consumers need the firewall's results without refitting it: the API, which
serves them to the console, and the submission ``.docx``. Both could have been
written by copying figures out of the document by hand. That is exactly how a
submission ends up quoting a recall the repo no longer produces — the model gets
retrained on the last night, the document does not, and the number on the slide
is from a run nobody can reproduce.

So the flow is one-way and mechanical::

    experiment -> report.write_results -> RESULTS.md -> results_doc.parse -> API / .docx

Re-running ``make firewall`` changes RESULTS.md, and every consumer changes with
it. Day 8 regenerates rather than retypes.

Why parse markdown rather than serialise the experiment
---------------------------------------------------------
Because RESULTS.md is the artefact a human reads and a judge is pointed at, so it
is the thing that must be true. A parallel JSON dump would be a second source of
truth, and a second source of truth is a source of drift: the moment the two
disagree, the interesting question becomes which one lied. Parsing the document
means the document cannot be stale relative to what is served — if a table is not
in RESULTS.md, no endpoint can invent it.

It also fails loudly. :func:`parse` raises on a document with no tables, which is
what a caller wants when ``make firewall`` has never been run: an empty results
screen with a clear error beats an empty results screen that looks like a zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from mantis.core.paths import REPO_ROOT

__all__ = ["ResultsDoc", "Section", "Table", "load", "parse"]

RESULTS_MD: Final[Path] = REPO_ROOT / "RESULTS.md"

_HEADING: Final[re.Pattern[str]] = re.compile(r"^(#{2,4})\s+(.*?)\s*$")

#: A markdown table separator: ``|---|---:|:--:|``. Used to recognise that the
#: line *above* was a header row rather than ordinary prose containing pipes.
_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"^\|[\s:|-]+\|$")


def _cells(line: str) -> list[str]:
    """Split one markdown table row into trimmed cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def strip_markup(text: str) -> str:
    """``**bold**``, ``*italic*`` and ``` `code` ``` down to their plain text.

    The console renders cells as text and python-docx writes them as runs;
    neither wants the asterisks. Applied to table cells only — section prose
    keeps its markup, because the .docx builder does render emphasis there.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    return text.replace("`", "").strip()


@dataclass(frozen=True, slots=True)
class Table:
    """One markdown table, as a header and rows of plain-text cells."""

    header: list[str]
    rows: list[list[str]]

    def as_records(self) -> list[dict[str, str]]:
        """Rows keyed by column name.

        Ragged rows are tolerated rather than raising: a hand-edit that drops a
        trailing cell should degrade one field, not blank the endpoint serving
        the whole table.
        """
        out: list[dict[str, str]] = []
        for row in self.rows:
            padded = row + [""] * (len(self.header) - len(row))
            out.append(dict(zip(self.header, padded[: len(self.header)], strict=True)))
        return out

    def column(self, name: str) -> list[str]:
        if name not in self.header:
            return []
        index = self.header.index(name)
        return [row[index] if index < len(row) else "" for row in self.rows]


@dataclass(slots=True)
class Section:
    """One ``##``/``###`` heading and everything under it, tables separated out."""

    level: int
    title: str
    prose: list[str] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The section's prose as one block, blank runs collapsed."""
        lines = [line for line in self.prose if line.strip()]
        return "\n\n".join(lines)

    @property
    def table(self) -> Table | None:
        """The first table, for the many sections that have exactly one."""
        return self.tables[0] if self.tables else None


@dataclass(slots=True)
class ResultsDoc:
    """``RESULTS.md``, parsed."""

    sections: list[Section]
    source: Path

    def find(self, needle: str) -> Section | None:
        """First section whose title contains ``needle``, case-insensitively.

        Matched on a substring rather than an exact title because the titles are
        prose written by ``report.py`` and are edited for readability; a lookup
        keyed on the exact string would break every time a heading gained a word.
        """
        lowered = needle.lower()
        for section in self.sections:
            if lowered in section.title.lower():
                return section
        return None

    def table_for(self, needle: str) -> Table | None:
        section = self.find(needle)
        return section.table if section else None

    @property
    def titles(self) -> list[str]:
        return [s.title for s in self.sections]


def parse(text: str, source: Path = RESULTS_MD) -> ResultsDoc:
    """Split a RESULTS.md-shaped document into sections and tables."""
    sections: list[Section] = [Section(level=1, title="preamble")]
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        heading = _HEADING.match(line)
        if heading:
            sections.append(Section(level=len(heading.group(1)), title=heading.group(2)))
            index += 1
            continue

        # A table is a row of pipes whose *next* line is a separator. Testing the
        # separator rather than the pipes is what keeps prose like "0.5 | 1.0"
        # from being read as a one-column table.
        if (
            line.strip().startswith("|")
            and index + 1 < len(lines)
            and _SEPARATOR.match(lines[index + 1].strip())
        ):
            header = [strip_markup(c) for c in _cells(line)]
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append([strip_markup(c) for c in _cells(lines[index])])
                index += 1
            sections[-1].tables.append(Table(header=header, rows=rows))
            continue

        sections[-1].prose.append(line)
        index += 1

    return ResultsDoc(sections=sections, source=source)


def load(path: Path = RESULTS_MD) -> ResultsDoc:
    """Parse ``RESULTS.md`` from disk.

    Raises loudly when the document is missing or carries no tables, because both
    mean ``make firewall`` has not been run and every number downstream would
    otherwise be silently empty.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist — run 'make firewall' (or 'python -m mantis.defense')"
        )
    doc = parse(path.read_text(encoding="utf-8"), source=path)
    if not any(section.tables for section in doc.sections):
        raise ValueError(f"{path} contains no tables; it is not a rendered results document")
    return doc


def main() -> None:
    """Print what the parser found. ``python -m mantis.defense.results_doc``."""
    doc = load()
    print(f"RESULTS.md parsed from {doc.source}")
    print(f"  {len(doc.sections)} sections, "
          f"{sum(len(s.tables) for s in doc.sections)} tables")
    print()
    for section in doc.sections:
        if section.title == "preamble" and not section.tables:
            continue
        mark = "#" * section.level
        print(f"  {mark:<5} {section.title}")
        for table in section.tables:
            print(f"           table {len(table.rows)}x{len(table.header)}: "
                  f"{', '.join(table.header[:5])}")


if __name__ == "__main__":
    main()
