pip install -U "huggingface_hub[cli]"
huggingface-cli download NovaSearch/stella_en_400M_v5 --local-dir ~/LLM/models/stella_en_400M_v5 --local-dir-use-symlinks False


# 2.  Infinity on Mac
docker run \
  --platform linux/amd64 \
  -p 8005:8005 \
  -v ~/LLM/models:/app/.cache \
  michaelf34/infinity:latest-cpu \
  v2 --model-id /app/.cache/USER-bge-m3 \
  --engine torch --device cpu --port 8005