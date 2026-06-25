import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        action="append",
        default=[],
        help="HuggingFace repo id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/root/autodl-tmp/hf-cache",
        help="Shared cache for HF_HOME and HUGGINGFACE_HUB_CACHE.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", args.cache_dir)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    for repo_id in args.repo_id:
        print(f"Downloading {repo_id} into {args.cache_dir}")
        snapshot_download(repo_id=repo_id, cache_dir=args.cache_dir, resume_download=True)


if __name__ == "__main__":
    main()
