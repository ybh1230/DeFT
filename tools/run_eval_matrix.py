import argparse
import os
import subprocess
from pathlib import Path


METHODS = {
    "baseline": "",
    "deft": "--deft",
    "deft_no_graph": "--deft --deft-graph-weight 0.0 --deft-local-weight 1.0",
    "deft_no_local": "--deft --deft-graph-weight 1.0 --deft-local-weight 0.0",
    "deft_no_multiscale": "--deft --deft-multiscale-strength 0.0",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a comparison/ablation matrix around an existing DeFT eval command."
    )
    parser.add_argument(
        "--base-cmd",
        required=True,
        help="Existing evaluation command without DeFT-specific flags or work-dir.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--work-root", default="./work_logs/deft_matrix")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/hf-cache")

    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        method_work_dir = str(work_root / method)
        cmd = f"{args.base_cmd} {METHODS[method]} --work-dir {method_work_dir}"
        print(f"\n[{method}] {cmd}")
        if not args.dry_run:
            subprocess.run(cmd, shell=True, cwd=args.workdir, env=env, check=True)


if __name__ == "__main__":
    main()
