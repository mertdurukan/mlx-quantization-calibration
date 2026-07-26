"""Build an arXiv-ready LaTeX submission package from paper.md.

paper.md is the single source. This script never edits it; it derives a LaTeX
document from it so the repository copy and the arXiv copy cannot drift apart.

Output (default build/arxiv/):
    main.tex                        pdflatex-compatible, standalone
    figure{1,2,3}_*.png             figures, flattened out of results/figures/
    main.pdf                        local compile, for proofreading only

arXiv does not accept a PDF that was produced from TeX (verified 2026-07-26,
info.arxiv.org/help/submit), so main.pdf is a local proof and main.tex plus the
figures are what gets uploaded.

Usage:
    python scripts/build_paper.py [--outdir build/arxiv] [--no-pdf]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PANDOC = shutil.which("pandoc") or "/opt/homebrew/bin/pandoc"
PDFLATEX = shutil.which("pdflatex") or str(
    Path.home() / "Library/TinyTeX/bin/universal-darwin/pdflatex"
)

# Unicode that pdflatex cannot typeset from a utf8 source without extra setup.
# Mapped in the LaTeX preamble instead of being stripped, so the text is
# unchanged -- see UNICODE_PREAMBLE.
UNICODE_PREAMBLE = r"""
\usepackage{textcomp}
\usepackage{newunicodechar}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{·}{\textperiodcentered}
\newunicodechar{§}{\S}
"""

# The result tables are 7 columns of "0.149 [0.125, 0.179]"; they do not fit at
# body size.
TABLE_PREAMBLE = r"""
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\scriptsize}
\setlength{\LTcapwidth}{\textwidth}
"""

# The prose cites long file paths in \texttt, which cannot be hyphenated by
# default and overflows the margin.
LAYOUT_PREAMBLE = r"""
\usepackage[htt]{hyphenat}
\emergencystretch=3em
\usepackage{float}
\floatplacement{figure}{!ht}
"""


def extract(md: str) -> dict[str, str]:
    """Split paper.md into title / byline / abstract / body."""
    lines = md.split("\n")

    if not lines[0].startswith("# "):
        raise SystemExit("paper.md must start with an H1 title line")
    title = lines[0][2:].strip()

    try:
        abstract_start = lines.index("## Abstract")
    except ValueError:
        raise SystemExit("paper.md must contain a '## Abstract' heading")

    byline = "\n".join(lines[1:abstract_start]).strip().strip("-").strip()

    rest = lines[abstract_start + 1 :]
    next_heading = next(
        (i for i, ln in enumerate(rest) if ln.startswith("## ")), None
    )
    if next_heading is None:
        raise SystemExit("no '## ' heading after the abstract; nothing to typeset")
    abstract = "\n".join(rest[:next_heading]).strip().strip("-").strip()
    body = "\n".join(rest[next_heading:])

    return {"title": title, "byline": byline, "abstract": abstract, "body": body}


def code_availability(byline: str) -> str:
    """Pull the code/data/DOI sentence out of the byline block.

    In a standalone PDF the reader cannot click 'this repository', so this ends
    up appended to the abstract where arXiv readers expect it.
    """
    m = re.search(r"\*\*Code and data:\*\*(.*?)(?=\n\*\*|\Z)", byline, re.S)
    if not m:
        raise SystemExit("byline block has no '**Code and data:**' entry")
    text = " ".join(m.group(1).split())
    # An <autolink> becomes \url{}, which xurl then breaks mid-word ("ht tps://")
    # inside the narrow abstract block. A labelled link hyphenates normally.
    text = re.sub(r"<(https?://([^>]+))>", r"[\2](\1)", text)
    return "Code and data: " + text


def author_line(byline: str) -> str:
    """Author line for the title block, taken from paper.md's byline.

    The byline block itself is not carried into the LaTeX body, so a
    correspondence address written only there would silently vanish from the PDF.
    Kept as markdown (pandoc parses metadata as markdown) rather than raw LaTeX,
    so the address becomes a real mailto link.
    """
    m = re.search(r"\*\*Author:\*\*\s*(.+)", byline)
    if not m:
        raise SystemExit("byline block has no '**Author:**' entry")
    author = m.group(1).strip()

    m = re.search(r"\*\*Correspondence:\*\*\s*<([^>]+)>", byline)
    if m:
        author += f" -- <{m.group(1).strip()}>"
    return author


def flatten_figures(body: str, outdir: Path) -> str:
    """Copy results/figures/*.png next to main.tex and rewrite the paths.

    arXiv is happier with a flat file list, and its filename charset does not
    include '/'.
    """
    copied = []

    def repl(m: re.Match) -> str:
        src = REPO / m.group(1)
        if not src.exists():
            raise SystemExit(f"figure referenced but missing: {src}")
        shutil.copy2(src, outdir / src.name)
        copied.append(src.name)
        return m.group(0).replace(m.group(1), src.name)

    body = re.sub(r"\((results/figures/[^)]+\.png)\)", repl, body)
    if len(copied) != 3:
        raise SystemExit(f"expected 3 figures, rewrote {len(copied)}: {copied}")

    # LaTeX numbers captions itself; the markdown alt text carries its own
    # "Figure N:" for readers of the .md, which would print twice.
    body, n = re.subn(r"!\[Figure \d+: ", "![", body)
    if n != 3:
        raise SystemExit(f"expected to strip 3 caption prefixes, stripped {n}")

    print(f"figures: {', '.join(copied)}")
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build/arxiv")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    outdir = (REPO / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    parts = extract((REPO / "paper.md").read_text())
    abstract = parts["abstract"] + "\n\n" + code_availability(parts["byline"])
    body = flatten_figures(parts["body"], outdir)

    # Horizontal rules are section separators in the markdown; in LaTeX they
    # render as stray horizontal lines.
    body = "\n".join(ln for ln in body.split("\n") if ln.strip() != "---")

    src = outdir / "body.md"
    src.write_text(body)
    (outdir / "abstract.md").write_text(abstract)
    header = outdir / "header.tex"
    header.write_text(UNICODE_PREAMBLE + TABLE_PREAMBLE + LAYOUT_PREAMBLE)

    cmd = [
        PANDOC,
        str(src),
        "--from=markdown",
        "--to=latex",
        "--standalone",
        "--shift-heading-level-by=-1",
        f"--include-in-header={header}",
        f"--metadata=title:{parts['title']}",
        "--metadata=date:July 2026",
        f"--metadata-file={_meta_file(outdir, abstract, author_line(parts['byline']))}",
        "--variable=documentclass:article",
        "--variable=fontsize:10pt",
        "--variable=geometry:margin=1in",
        "--variable=colorlinks:true",
        "--variable=linkcolor:blue",
        "--variable=urlcolor:blue",
        "--output=" + str(outdir / "main.tex"),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {outdir / 'main.tex'}")

    write_form_abstract(outdir, abstract)
    make_tarball(outdir)

    if args.no_pdf:
        return 0

    for i in (1, 2):
        r = subprocess.run(
            [PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=outdir,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            tail = "\n".join(r.stdout.strip().split("\n")[-40:])
            print(tail, file=sys.stderr)
            raise SystemExit(f"pdflatex failed on pass {i}")
    print(f"wrote {outdir / 'main.pdf'}")
    return 0


# arXiv's abstract form field is a plain-text box; markdown markup and typographic
# unicode both survive badly there.
ARXIV_ABSTRACT_MAX = 1920

ASCII_FOLD = {
    "—": "--", "–": "-", "×": "x", "·": "-", "§": "Section ",
    "≥": ">=", "≤": "<=", "Δ": "Delta ", "’": "'", "“": '"',
    "”": '"',
}


def write_form_abstract(outdir: Path, abstract: str) -> Path:
    """Plain-ASCII abstract for pasting into arXiv's submission form."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", abstract)   # links -> label only
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = t.replace("`", "")
    for k, v in ASCII_FOLD.items():
        t = t.replace(k, v)
    t = re.sub(r"\s+", " ", t).strip()

    leftover = sorted({c for c in t if ord(c) > 127})
    if leftover:
        raise SystemExit(f"non-ASCII left in form abstract: {leftover}")

    # "abstracts longer than 1920 characters will not be accepted"
    # (info.arxiv.org/help/prep, verified 2026-07-26). Fail here rather than at
    # submission time, when the paper is otherwise ready to go.
    if len(t) > ARXIV_ABSTRACT_MAX:
        raise SystemExit(
            f"form abstract is {len(t)} chars, over arXiv's {ARXIV_ABSTRACT_MAX} limit "
            f"by {len(t) - ARXIV_ABSTRACT_MAX}; shorten the abstract in paper.md"
        )

    p = outdir / "abstract-for-arxiv-form.txt"
    p.write_text(t + "\n")
    print(f"wrote {p} ({len(t.split())} words)")
    return p


def make_tarball(outdir: Path) -> Path:
    """Pack exactly what arXiv should receive: main.tex plus the three figures.

    Deliberately excludes main.pdf -- uploading a PDF alongside TeX source makes
    arXiv ambiguous about which to serve -- and the pandoc intermediates.
    """
    import tarfile

    members = ["main.tex"] + sorted(p.name for p in outdir.glob("figure*.png"))
    tar = outdir / "arxiv-submission.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for name in members:
            t.add(outdir / name, arcname=name)
    print(f"wrote {tar} ({len(members)} files: {', '.join(members)})")
    return tar


def _meta_file(outdir: Path, abstract: str, author: str) -> Path:
    """YAML metadata file: the abstract is too long to pass via --metadata, and
    the author line needs markdown parsing (for the mailto autolink)."""

    def q(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    p = outdir / "meta.yaml"
    p.write_text(
        "---\n"
        f'abstract: "{q(abstract)}"\n'
        f'author: "{q(author)}"\n'
        "---\n"
    )
    return p


if __name__ == "__main__":
    raise SystemExit(main())
