#!/bin/bash
# Local helper: bake Whisper weights into ./models (for dev).
# Docker/Cog bake uses cog.yaml → /models (outside COPY . /src). See docs/TESTING.md / README.
# Default: large-v3-turbo only. Override: WHISPER_BAKE_MODELS=all

set -euo pipefail

WHISPER_BAKE_MODELS="${WHISPER_BAKE_MODELS:-large-v3-turbo}"
MODELS_ROOT="${MODELS_ROOT:-./models}"

download() {
  local file_url="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget -O "$destination_path" "$file_url"
  else
    echo "$destination_path already exists. No need to download."
  fi
}

bake_faster_whisper() {
  local name="$1"
  local hf_repo="$2"
  local dir="${MODELS_ROOT}/faster-whisper-${name}"
  mkdir -p "$dir"
  download "https://huggingface.co/${hf_repo}/resolve/main/config.json" "$dir/config.json"
  download "https://huggingface.co/${hf_repo}/resolve/main/model.bin" "$dir/model.bin"
  download "https://huggingface.co/${hf_repo}/resolve/main/preprocessor_config.json" "$dir/preprocessor_config.json"
  download "https://huggingface.co/${hf_repo}/resolve/main/tokenizer.json" "$dir/tokenizer.json"
  download "https://huggingface.co/${hf_repo}/resolve/main/vocabulary.json" "$dir/vocabulary.json"
}

bake_whisper_model() {
  case "$1" in
    tiny) bake_faster_whisper "tiny" "Systran/faster-whisper-tiny" ;;
    large-v3) bake_faster_whisper "large-v3" "Systran/faster-whisper-large-v3" ;;
    large-v3-turbo)
      bake_faster_whisper "large-v3-turbo" "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
      ;;
    all)
      bake_whisper_model tiny
      bake_whisper_model large-v3
      bake_whisper_model large-v3-turbo
      ;;
    *)
      echo "Unknown WHISPER_BAKE_MODELS entry: $1 (expected tiny, large-v3, large-v3-turbo, or all)"
      exit 1
      ;;
  esac
}

if [ "$WHISPER_BAKE_MODELS" = "all" ]; then
  bake_whisper_model all
else
  IFS=',' read -ra _models <<< "$WHISPER_BAKE_MODELS"
  for _model in "${_models[@]}"; do
    _model="${_model// /}"
    [ -n "$_model" ] || continue
    bake_whisper_model "$_model"
  done
fi

vad_model_dir="${MODELS_ROOT}/vad"
mkdir -p "$vad_model_dir"
download "$(python3 ./get_vad_model_url.py)" "$vad_model_dir/whisperx-vad-segmentation.bin"

echo "Baked models under ${MODELS_ROOT} (WHISPER_BAKE_MODELS=${WHISPER_BAKE_MODELS})"
