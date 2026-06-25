import argparse
import csv
from datetime import datetime
from pathlib import Path


FIELDS = [
    "time",
    "dataset",
    "method",
    "backbone",
    "seed",
    "mIoU",
    "mAcc",
    "boundary_F",
    "small_object_mIoU",
    "fps",
    "gpu",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    for field in FIELDS[1:]:
        parser.add_argument(f"--{field}", default="")
    parser.add_argument("--file", default="results/results_log.csv")
    args = parser.parse_args()

    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    row = {field: getattr(args, field, "") for field in FIELDS[1:]}
    row["time"] = datetime.now().isoformat(timespec="seconds")

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Recorded result in {path}")


if __name__ == "__main__":
    main()
