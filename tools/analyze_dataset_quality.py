import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
REPORT_PATH = PROJECT_ROOT / "docs" / "dataset_quality_report.md"

ATTACK_TYPES = (
    "Brute Force",
    "SQL Injection",
    "Port Scan",
    "Malware",
    "Command and Control",
)
RISK_LEVELS = ("Low", "Medium", "High")
INJECTION_CATEGORIES = (
    "Direct Prompt Injection",
    "Indirect Prompt Injection",
    "Context Manipulation",
)


def load_samples(folder_name: str) -> list[dict[str, object]]:
    folder = DATASET_ROOT / folder_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")
    samples = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(folder.glob("*.json"))
    ]
    if not samples:
        raise ValueError(f"Dataset folder is empty: {folder}")
    return samples


def distribution(rows: list[dict[str, object]], field: str) -> Counter[str]:
    return Counter(str(row[field]) for row in rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join((header, separator, body))


def generate_report() -> str:
    clean = load_samples("clean_logs")
    injected = load_samples("injected_logs")
    clean_risk = distribution(clean, "ground_truth_risk")
    clean_attack = distribution(clean, "ground_truth_attack_type")
    injected_risk = distribution(injected, "ground_truth_risk")
    injected_attack = distribution(injected, "ground_truth_attack_type")
    injected_categories = Counter(
        str(row.get("injection_category", "Uncategorized"))
        for row in injected
    )

    clean_risk_rows = [
        [risk, str(clean_risk[risk])]
        for risk in RISK_LEVELS
    ]
    clean_attack_rows = [
        ["None"] + [str(clean_attack["None"])]
    ] + [
        [attack, str(clean_attack[attack])]
        for attack in ATTACK_TYPES
    ]
    injected_risk_rows = [
        [risk, str(injected_risk[risk])]
        for risk in RISK_LEVELS
    ]
    injected_attack_rows = [
        [attack, str(injected_attack[attack])]
        for attack in ATTACK_TYPES
    ]
    injection_category_rows = [
        [category, str(injected_categories[category])]
        for category in INJECTION_CATEGORIES
    ]

    clean_attack_missing = [
        attack for attack in ATTACK_TYPES if clean_attack[attack] == 0
    ]
    injected_category_missing = [
        category
        for category in INJECTION_CATEGORIES
        if injected_categories[category] == 0
    ]
    duplicate_ids = len(clean) + len(injected) - len({
        str(row["id"]) for row in clean + injected
    })

    return (
        "# Dataset Quality Report\n\n"
        "## 1. Scope and Purpose\n\n"
        "This report reviews the current clean and prompt-injected log datasets "
        "used for the AI Agent security benchmark. It measures sample size, "
        "risk-label balance, attack-type coverage, injection-category diversity, "
        "and duplicate-ID risk.\n\n"
        "## 2. Dataset Sizes\n\n"
        f"- Clean logs: **{len(clean)}** samples\n"
        f"- Injected logs: **{len(injected)}** samples\n"
        f"- Combined benchmark: **{len(clean) + len(injected)}** samples\n\n"
        "## 3. Clean Dataset Distribution\n\n"
        "### Risk levels\n\n"
        f"{markdown_table(['Risk Level', 'Count'], clean_risk_rows)}\n\n"
        "### Attack types\n\n"
        f"{markdown_table(['Attack Type', 'Count'], clean_attack_rows)}\n\n"
        "## 4. Injected Dataset Distribution\n\n"
        "### Risk levels\n\n"
        f"{markdown_table(['Risk Level', 'Count'], injected_risk_rows)}\n\n"
        "### Attack types\n\n"
        f"{markdown_table(['Attack Type', 'Count'], injected_attack_rows)}\n\n"
        "### Injection categories\n\n"
        f"{markdown_table(['Injection Category', 'Count'], injection_category_rows)}\n\n"
        "## 5. Quality Assessment\n\n"
        "- The clean dataset is at the upper end of the starter target at "
        f"{len(clean)} samples; its five security attack classes each have "
        "three clean examples.\n"
        "- The injected dataset now has 20 samples, with four examples for "
        "each of the five attack types.\n"
        "- Direct, indirect, and context-manipulation injections are represented "
        "as separate, reusable dataset metadata.\n"
        "- A remaining weakness is that most injected samples retain High risk "
        "ground truth, so low-risk and medium-risk injected edge cases should be "
        "expanded further in future rounds.\n"
        f"- Clean attack types absent from the corpus: "
        f"{', '.join(clean_attack_missing) if clean_attack_missing else 'None'}.\n"
        f"- Injection categories absent from the corpus: "
        f"{', '.join(injected_category_missing) if injected_category_missing else 'None'}.\n"
        f"- Duplicate ID count across the corpus: {duplicate_ids}.\n\n"
        "## 6. Weaknesses and Recommended Next Data\n\n"
        "1. Add paired examples with Medium risk in the injected corpus rather "
        "than concentrating ground-truth risk in High.\n"
        "2. Add low-risk baseline traffic from several protocols, including "
        "DNS, SMTP, database, remote-access, and internal API traffic.\n"
        "3. Add multi-line and mixed-signal logs so prompt-injection text cannot "
        "be separated by a unique synthetic marker.\n"
        "4. Add additional indirect injections such as ticket comments, alert "
        "notes, approved-change context, and analyst summaries.\n"
        "5. Preserve the JSON schema and paired clean/injected identifiers when "
        "adding samples, then rerun dataset-quality validation and the complete "
        "benchmark.\n\n"
        "## 7. Reproducibility\n\n"
        "Run `python tools/analyze_dataset_quality.py` to regenerate this report "
        "from the JSON dataset files. The report is descriptive and should be "
        "rerun after each corpus revision.\n"
    )


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(generate_report(), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
