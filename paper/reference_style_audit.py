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
    args = parser.parse_args()
    rows = [analyze(identifier, args.archive_dir / f"{identifier}.tar") for identifier in PAPERS]
    args.json.write_text(json.dumps({"papers": rows, "perplexity_targeting_used": False}, indent=2, sort_keys=True) + "\n")
    lines = ["# Pre-2024 paper structure audit", "", "This audit uses source-level section lengths and float conventions. Perplexity targeting and author imitation are not used.", ""]
    for row in rows:
        lines += [f"## {row['paper']}", "", f"Abstract: {row['abstract_words']} words. Main sections: {row['main_text_words']} words. Figures: {row['figures']} ({row['double_column_figures']} double-column). Tables: {row['tables']} ({row['double_column_tables']} double-column).", ""]
        lines += [f"- {section['title']}: {section['words']} words" for section in row["sections"]] + [""]
    args.markdown.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
