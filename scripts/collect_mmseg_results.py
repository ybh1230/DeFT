import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"aAcc:\s*([0-9.]+)\s+mIoU:\s*([0-9.]+)\s+mAcc:\s*([0-9.]+)"
)


def scan_file(path: Path):
    text = path.read_text(errors="ignore")
    matches = PATTERN.findall(text)
    if not matches:
        return None
    aacc, miou, macc = matches[-1]
    return {"aAcc": float(aacc), "mIoU": float(miou), "mAcc": float(macc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", default="SFP_ICCV/work_logs")
    parser.add_argument("--out", default="results/collected_results.json")
    args = parser.parse_args()

    root = Path(args.work_root)
    results = {}
    for log in root.rglob("*.log"):
        item = scan_file(log)
        if item:
            results[str(log.relative_to(root))] = item

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
