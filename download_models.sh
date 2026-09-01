#!/bin/bash

# ==============================================================================
# SLM Model Downloader - EdgeAI Provider
# ==============================================================================
# This script automates the download of GGUF models from Hugging Face.
# It uses the 'hf' CLI tool.
# ==============================================================================

# --- Configuration ---
MODELS_DIR="/config/REPOS/providerZinho/models"
# Format: "Model_ID|Repo_Owner/Repo_Name|File_Name"
MODELS=(
    "qwen-0.5b|Qwen/Qwen2.5-0.5B-Instruct-GGUF|qwen2.5-0.5b-instruct-q6_k.gguf"
    "llama-3.2-3b|Bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-IQ3_M.gguf"
    "gemma-2b|Bartowski/gemma-2-2b-it-GGUF|gemma-2-2b-it-Q4_K_M.gguf"
    "phi-3.5-mini|Bartowski/Phi-3.5-mini-instruct-GGUF|Phi-3.5-mini-instruct-Q4_K_M.gguf"
)

# Ensure models directory exists
mkdir -p "$MODELS_DIR"

echo "🚀 Starting SLM Fleet Download..."
echo "📂 Destination: $MODELS_DIR"
echo "----------------------------------------------------------------"

for entry in "${MODELS[@]}"; do
    # Split the entry by pipe
    IFS="|" read -r MODEL_ID REPO FILE <<< "$entry"
    
    echo "📦 Downloading [$MODEL_ID] from $REPO..."
    
    # Execute download
    hf download "$REPO" "$FILE" --local-dir "$MODELS_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded $MODEL_ID"
    else
        echo "❌ Failed to download $MODEL_ID. Check if the repo exists or if you need to accept terms on HF."
    fi
    echo "----------------------------------------------------------------"
done

echo "🏁 Download process finished."
