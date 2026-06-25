#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

mkdir -p "${HF_HOME}"

python tools/predownload_hf.py \
  --repo-id openai/clip-vit-base-patch16 \
  --repo-id openai/clip-vit-large-patch14 \
  --cache-dir "${HF_HOME}"
