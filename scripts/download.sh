#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container or submitting compute jobs.
# Idempotent: files that already exist are skipped.
#
# Estimated total download size: 3.1 GB (Qwen2.5-1.5B-Instruct weights and tokenizer)
# Estimated disk usage after extraction: 3.1 GB
#
# The model is only needed to (re)build the language-model token trees of the
# lm_consensus_search experiment (python -m data.lm_consensus.build_trees, GPU recommended).
# The trees themselves are committed (data/lm_consensus/trees/ and, held out,
# eval/test/lm_trees/), so evaluation and the reference solver need no download.
# The synthetic core-test experiments need no external data.

set -e

cd /home/user

MODEL_DIR=data/downloads/Qwen2.5-1.5B-Instruct
BASE_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main
mkdir -p "$MODEL_DIR"
for f in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json LICENSE model.safetensors; do
    if [ ! -s "$MODEL_DIR/$f" ]; then
        echo "fetching $f"
        curl -L --fail --retry 3 -o "$MODEL_DIR/$f.part" "$BASE_URL/$f"
        mv "$MODEL_DIR/$f.part" "$MODEL_DIR/$f"
    fi
done
echo "model files in $MODEL_DIR"
