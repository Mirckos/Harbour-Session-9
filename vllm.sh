git clone https://github.com/vllm-project/vllm.git
cd vllm

# ARM-var
DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  -f docker/Dockerfile.arm \
  --build-arg max_jobs=1 \
  -t vllm-openai-cpu-arm \
  --memory 12g --memory-swap 16g \
  --shm-size 4g .

docker run --rm \
  -p 8000:8000 \
  --shm-size=4g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-openai-cpu-arm \
  --model /root/models/Qwen2-0.5B-Instruct \
  --dtype float16
