import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_map(tensor: torch.Tensor, path: Path, title: str) -> None:
    arr = tensor.detach().float().cpu()
    while arr.dim() > 2:
        arr = arr[0]
    plt.figure(figsize=(4, 4))
    plt.imshow(arr.numpy(), cmap="viridis")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.pt"))[: args.limit]
    for file in files:
        maps = torch.load(file, map_location="cpu")
        for name, tensor in maps.items():
            save_map(tensor, output_dir / f"{file.stem}_{name}.png", name)

    print(f"Saved {len(files)} DeFT map groups to {output_dir}")


if __name__ == "__main__":
    main()
