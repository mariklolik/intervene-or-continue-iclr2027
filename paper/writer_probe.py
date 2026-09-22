import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "paper")
from reference_style_audit import PAPERS, archive_text, expand

HEDGES = ("may", "might", "could", "suggest", "appear", "seem", "likely", "possibly", "perhaps")
FIRST = ("we ", "our ", "we'", "us ")


def clean(text):
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(cite[a-z]*|ref|label|footnote|textsc|textbf|textit|emph|texttt)\*?(\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    return re.sub(r"[{}~\\]", " ", text)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean(text)) if len(s.split()) > 3]


def profile(name, text):
    sents = sentences(text)
    words = [len(s.split()) for s in sents]
    lower = " ".join(sents).lower()
    paras = [p for p in re.split(r"\n\s*\n", text) if len(clean(p).split()) > 15]
    return {
        "name": name, "sentences": len(sents),
        "median_words": sorted(words)[len(words) // 2] if words else 0,
        "long_share": round(sum(w > 35 for w in words) / max(len(words), 1), 3),
        "hedge_per_100": round(100 * sum(lower.count(h) for h in HEDGES) / max(len(sents), 1), 1),
        "first_person_per_100": round(100 * sum(lower.count(f) for f in FIRST) / max(len(sents), 1), 1),
        "paragraphs": len(paras),
        "para_openings": [" ".join(clean(p).split()[:12]) for p in paras[:4]],
    }


def main():
    root = Path(sys.argv[1])
    out = {}
    for paper, main_file in PAPERS.items():
        files = archive_text(root / f"{paper}.tar")
        body = expand(main_file, files)
        blocks = re.split(r"\\section\*?\{([^}]*)\}", body)
        out[paper] = [profile(title, chunk) for title, chunk in zip(blocks[1::2], blocks[2::2])]
    manuscript = Path("paper/main.tex").read_text()
    blocks = re.split(r"\\section\*?\{([^}]*)\}", manuscript)
    out["manuscript"] = [profile(title, chunk) for title, chunk in zip(blocks[1::2], blocks[2::2])]
    for name in ("results", "r2_results", "analysis", "r2_analysis", "r2_transfer"):
        path = Path("paper/generated") / f"{name}.tex"
        if path.exists():
            out["manuscript"].append(profile(f"generated/{name}", path.read_text()))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
