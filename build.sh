#!/bin/bash

set -e

download() {
  local file_url="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget -O "$destination_path" "$file_url"
  else
      echo "$destination_path already exists. No need to download."
  fi
}

# faster-whisper-tiny (Systran/faster-whisper-tiny)
faster_whisper_tiny_dir=models/faster-whisper-tiny
mkdir -p $faster_whisper_tiny_dir
download "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/config.json" "$faster_whisper_tiny_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin" "$faster_whisper_tiny_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/preprocessor_config.json" "$faster_whisper_tiny_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/tokenizer.json" "$faster_whisper_tiny_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/vocabulary.json" "$faster_whisper_tiny_dir/vocabulary.json"

# faster-whisper-large-v3 (Systran/faster-whisper-large-v3)
faster_whisper_large_v3_dir=models/faster-whisper-large-v3
mkdir -p $faster_whisper_large_v3_dir
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json" "$faster_whisper_large_v3_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin" "$faster_whisper_large_v3_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_large_v3_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json" "$faster_whisper_large_v3_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json" "$faster_whisper_large_v3_dir/vocabulary.json"

# large-v3-turbo (mobiuslabsgmbh/faster-whisper-large-v3-turbo) — faster, less VRAM
faster_whisper_turbo_dir=models/faster-whisper-large-v3-turbo
mkdir -p $faster_whisper_turbo_dir
download "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/config.json" "$faster_whisper_turbo_dir/config.json"
download "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/model.bin" "$faster_whisper_turbo_dir/model.bin"
download "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/preprocessor_config.json" "$faster_whisper_turbo_dir/preprocessor_config.json"
download "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/tokenizer.json" "$faster_whisper_turbo_dir/tokenizer.json"
download "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/main/vocabulary.json" "$faster_whisper_turbo_dir/vocabulary.json"

pip install -U whisperx

vad_model_dir=models/vad
mkdir -p $vad_model_dir

download $(python3 ./get_vad_model_url.py) "$vad_model_dir/whisperx-vad-segmentation.bin"

cog run python