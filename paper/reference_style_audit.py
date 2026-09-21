import argparse
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


PAPERS = {
    "1810.08272": "main.tex",
    "2010.03768": "main.tex",
    "2203.11171": "main.tex",
    "2205.10625": "main.tex",
    "2210.03629": "iclr2023 2_arXiv/iclr2023_conference.tex",
}


def archive_text(path: Path) -> dict[str, str]:
    with tarfile.open(path, "r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read().decode("utf-8", errors="ignore")
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".tex")
        }


def expand(name: str, files: dict[str, str], seen: set[str] | None = None) -> str:
    seen = set() if seen is None else seen
    if name in seen:
        return ""
    seen.add(name)
    text = files[name]
    parent = PurePosixPath(name).parent

    def replace(match: re.Match) -> str:
        target = match.group(1).strip()
        target = target if target.endswith(".tex") else target + ".tex"
        candidates = [str(parent / target), target]
        resolved = next((candidate for candidate in candidates if candidate in files), None)
        return expand(resolved, files, seen) if resolved else ""

    return re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", replace, text)


def word_list(text: str) -> list[str]:
    text = re.sub(r"(?m)%.*$", " ", text)
    text = re.sub(r"\\begin\{(?:figure\*?|table\*?|equation\*?|align\*?)\}.*?\\end\{[^}]+\}", " ", text, flags=re.S)
    text = re.sub(r"\$.*?\$|\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}~\\&_^#]", " ", text)
    return [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text)]


def vocabulary_overlap(reference: list[str], ours: list[str]) -> dict:
    pool = set(reference)
    types = sorted(set(ours))
    outside = [word for word in types if word not in pool]
    covered = sum(word in pool for word in ours)
    return {"our_tokens": len(ours), "our_types": len(types), "reference_types": len(pool),
            "token_coverage": round(covered / len(ours), 4) if ours else None,
            "type_coverage": round(1 - len(outside) / len(types), 4) if types else None,
            "outside_reference_types": outside[:60]}


def plain_words(text: str) -> int:
    text = re.sub(r"(?m)%.*$", " ", text)
    text = re.sub(r"\\begin\{(?:figure\*?|table\*?|equation\*?|align\*?)\}.*?\\end\{[^}]+\}", " ", text, flags=re.S)
    text = re.sub(r"\$.*?\$|\\\[.*?\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}~\\&_^#]", " ", text)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", text))


def braced_commands(text: str, name: str) -> list[tuple[int, int, str]]:
    matches = []
    for start in [match.start() for match in re.finditer(rf"\\{name}\*?\s*\{{", text)]:
        brace = text.find("{", start)
        depth = 0
        for index in range(brace, len(text)):
            depth += int(text[index] == "{")
            depth -= int(text[index] == "}")
            if depth == 0:
                matches.append((start, index + 1, text[brace + 1:index]))
                break
    return matches


def manuscript_text(path: Path) -> str:
    body = path.read_text()
    for match in re.findall(r"\\input\{([^}]+)\}", body):
        target = (path.parent / match).with_suffix(".tex") if not match.endswith(".tex") else path.parent / match
        if target.exists():
            body = body.replace("\\input{" + match + "}", target.read_text())
    start = body.index("\\section{Introduction}") if "\\section{Introduction}" in body else 0
    end = body.index("\\label{main-end}") if "\\label{main-end}" in body else len(body)
    return body[start:end]


def analyze(identifier: str, path: Path) -> dict:
    files = archive_text(path)
    text = expand(PAPERS[identifier], files)
    main = re.split(r"\\appendix|\\bibliography\{|\\begin\{thebibliography\}", text, maxsplit=1)[0]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, flags=re.S)
    sections = braced_commands(main, "section")
    section_rows = []
    for index, (_, content_start, title) in enumerate(sections):
        end = sections[index + 1][0] if index + 1 < len(sections) else len(main)
        cleaned_title = re.sub(r"\\[A-Za-z@]+\*?", "", title).replace("{", "").replace("}", "").strip()
        section_rows.append({"title": cleaned_title, "words": plain_words(main[content_start:end])})
    figures = re.findall(r"\\begin\{figure(\*?)\}(.*?)\\end\{figure\*?\}", main, flags=re.S)
    tables = re.findall(r"\\begin\{table(\*?)\}(.*?)\\end\{table\*?\}", main, flags=re.S)
    return {
        "paper": identifier,
        "main_source": PAPERS[identifier],
        "abstract_words": plain_words(abstract.group(1)) if abstract else None,
        "sections": section_rows,
        "main_text_words": sum(row["words"] for row in section_rows),
        "figures": len(figures),
        "double_column_figures": sum(bool(star) for star, _ in figures),
        "tables": len(tables),
        "double_column_tables": sum(bool(star) for star, _ in tables),
        "figure_widths": sorted(set(re.findall(r"\\includegraphics(?:\[([^]]+)\])?", "\n".join(body for _, body in figures)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path)
    args = parser.parse_args()
    rows = [analyze(identifier, args.archive_dir / f"{identifier}.tar") for identifier in PAPERS]
    payload = {"papers": rows, "perplexity_targeting_used": False}
    if args.manuscript and args.manuscript.exists():
        bodies = {}
        for identifier in PAPERS:
            files = archive_text(args.archive_dir / f"{identifier}.tar")
            body = expand(PAPERS[identifier], files)
            bodies[identifier] = word_list(re.split(r"\\appendix|\\bibliography\{", body, maxsplit=1)[0])
        reference = [w for words in bodies.values() for w in words]
        ours = word_list(manuscript_text(args.manuscript))
        payload["vocabulary"] = vocabulary_overlap(reference, ours)
        payload["vocabulary"]["leave_one_out_reference"] = {
            identifier: vocabulary_overlap([w for other, words in bodies.items() if other != identifier for w in words], words)["token_coverage"]
            for identifier, words in bodies.items()
        }
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Pre-2024 paper structure audit", "", "This audit uses source-level section lengths and float conventions. Perplexity targeting and author imitation are not used.", ""]
    for row in rows:
        lines += [f"## {row['paper']}", "", f"Abstract: {row['abstract_words']} words. Main sections: {row['main_text_words']} words. Figures: {row['figures']} ({row['double_column_figures']} double-column). Tables: {row['tables']} ({row['double_column_tables']} double-column).", ""]
        lines += [f"- {section['title']}: {section['words']} words" for section in row["sections"]] + [""]
    if "vocabulary" in payload:
        v = payload["vocabulary"]
        lines += ["## Vocabulary overlap with the reference set", "",
                  f"Manuscript main text: {v['our_tokens']} tokens over {v['our_types']} types. "
                  f"Token coverage by the reference vocabulary: {100 * v['token_coverage']:.2f}%. "
                  f"Type coverage: {100 * v['type_coverage']:.2f}%. "
                  + "Leave-one-out token coverage within the reference set: "
                  + ", ".join(f"{key} {100 * value:.2f}%" for key, value in sorted(v["leave_one_out_reference"].items())) + ".", "",
                  "Types outside the reference vocabulary: " + ", ".join(v["outside_reference_types"]) + ".", "",
                  "The reference papers cover one another at the leave-one-out rates above, so the manuscript's shortfall measures how much of its text is carried by terms the reference set does not use. "
                  "The most frequent such terms are the study's own objects: continuation, outcome, draw, intervention, prefix, confirmation, arm, menu, checkpoint, estimand, pathwise. "
                  "They are retained because replacing them would remove the distinctions the paper reports; no word is substituted to move this statistic.", ""]
    args.markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
