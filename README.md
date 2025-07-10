# Week 9 – LLMOps

### vLLM – Inference Server (Docker)

Loading the instruct-tuned Qwen model:

```bash
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct ~/LLM/models/Qwen2-0.5B-Instruct
```

Deploy the model as an OpenAI-compatible server:


```bash
docker run --runtime nvidia --gpus all \
    -v ~/LLM/models/:/root/models \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --trust-remote-code \
    --model /root/models/Qwen2-0.5B-Instruct
```

*The complete list of vLLM engine launch arguments can be found [here](https://docs.vllm.ai/en/latest/serving/engine_args.html).*

Network access:

```python
from openai import OpenAI

client = OpenAI(api_key="dummy_key", base_url="http://127.0.0.1:8000/v1")

prompt = "Tell me about yourself"

payload = {
    "model": "/root/models/Qwen2-0.5B-Instruct",
    "messages": [{"role": "user", "content": prompt}],
}

response = client.chat.completions.create(**payload)
# We’re already using chat.completions; in the previous example it would have been completions. 

if response:
    generated_text = response.choices[0].message.content.strip()
    print(f"Generated text: {generated_text}")
else:
    print("Failed to generate text.")
```

Chat modelling (there are three principal roles – `system`, `assistant`, and `user`. Not every model supports the `system` role; the only way to know is through hands-on testing, reading source code, community feedback, etc. For example, Gemma 2 did **not** support the `system` role):

```python
payload = {
    "model": "Qwen2-0.5B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a legend"},
        {"role": "assistant", "content": "Are you teasing your sisters with examples?"},
        {"role": "user", "content": "What am I doing?"},
        {"role": "assistant", "content": "Everyone hears, but you don’t hear?"},
        {"role": "user", "content": "Whom?"},
        {"role": "assistant", "content": "THE SIIIIS-TERS!"}
    ],
}

response = client.chat.completions.create(**payload)

if response:
    generated_text = response.choices[0].message.content.strip()
    print(f"Generated text: {generated_text}")
else:
    print("Failed to generate text.")
```

Suppose our GPU resources are limited, or we have none at all (vLLM itself can run quantised models and save resources, but with the next tool you can push things to the extreme and deploy literally on a toaster). For reference, see [Quantisation in vLLM](https://docs.vllm.ai/en/latest/features/quantization/index.html): you can run models already quantised by others, quantise *on the fly* via vLLM (see bitsandbytes), or even run “foreign” formats such as GGUF, native to **llama.cpp**, which we look at next.

## llama.cpp – Inference Server (Docker)

There is **llama-cpp-python**, but I strongly advise against using it. It is just a Python wrapper around llama.cpp that always lags behind and, from experience, is riddled with issues.

*Documentation:* [https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)

Choose a model (e.g. [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)) and download it (for a direct link, click on the file in the [repository](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main), “Files and versions” tab):

```bash
cd ~/LLM/models
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The convenience of **llama.cpp** is that a single file contains everything needed for a basic run.

Launch the server container in detached mode:

```bash
sudo docker run -d --runtime nvidia --gpus all --restart unless-stopped -p 8000:8000 \
-v ~/LLM/models/:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
-m /models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
--parallel 1 --port 8000 --verbose -fa
```

*Launch arguments* are documented [here](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md).
I suspect `-fa` (flash-attention) does not work, or not for every model.
The number of simultaneous requests is controlled with

```
-np, --parallel N
```

Note that the context-window size

```
-c, --ctx-size N
```

is divided by the number of threads. In other words, multiply the context window by **N** for realistic capacity.

Observe the dramatic difference in memory usage. If you need to run on CPU only, use the image `llama.cpp:server`.

Network access:

```python
from openai import OpenAI

client = OpenAI(api_key="dummy_key", base_url="http://127.0.0.1:8000")

prompt = "Hi, how are you?"

payload = {
    "model": "llama.cpp",  # You can put anything here; llama.cpp doesn’t check it.
    "messages": [{"role": "user", "content": prompt}],
}

response = client.chat.completions.create(**payload)

if response:
    generated_text = response.choices[0].message.content.strip()
    print(f"Generated text: {generated_text}")
else:
    print("Failed to generate text.")
```

## infinity

A tool for hosting **Sentence Transformers**; primarily used to vectorise text (feature extraction over text objects or building RAG systems).

*Documentation:* [https://github.com/michaelfeil/infinity](https://github.com/michaelfeil/infinity)

To pick an embedding model for a particular language, consult the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard_legacy).

Model choice:

* [USER-bge-m3](https://huggingface.co/deepvk/USER-bge-m3)

Download and launch – we’re seasoned now:

```bash
git clone https://huggingface.co/deepvk/USER-bge-m3 ~/LLM/models/USER-bge-m3
sudo docker run -d --gpus all -v ~/LLM/models/:/app/.cache -p 8005:8005 michaelf34/infinity:latest v2 \
--model-id /app/.cache/USER-bge-m3 \
--port 8005
```

Network access:

```python
from openai import OpenAI  # or AsyncOpenAI
from datetime import datetime
from httpx import Client  # or AsyncClient

api_base = "http://127.0.0.1:8005"
api_key = "dummy_key"

client_emb = OpenAI(api_key=api_key, base_url=api_base)

models_list = [m.id for m in client_emb.models.list().data]

print(datetime.now())

for model in models_list:
    responses = client_emb.embeddings.create(
        input=[
            "Hello everyone, we’re from the Central University",
            "This week we’re deploying various large language models",
        ],
        model=model,
    )
    for data in responses.data:
        print(data.embedding[:5])
        print(len(data.embedding))
        print(model)

print(datetime.now())
```

You can also serve models for **re-ranking** and **classification** at the `/rerank` and `/classify` endpoints respectively, but you need models that support those tasks. Examples (are you *teasing your sisters* again?):

```
rerank:
https://huggingface.co/BAAI/bge-reranker-v2-m3

classify:
https://huggingface.co/sismetanin/rubert-ru-sentiment-rusentiment
```

*It seems OpenAI no longer provides wrappers for these methods, so you’ll have to resort to `python-requests`.*

Complete API description: [https://michaelfeil-infinity.hf.space/docs](https://michaelfeil-infinity.hf.space/docs)

**It might feel odd at first, but all these model calls through OpenAI are merely convenient wrappers around plain HTTP requests.**
Here is what a chat looks like in ancient `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer dummy-key" \
-d '{
"model": "/root/models/Qwen2-0.5B-Instruct",
"messages": [
  {
    "role": "system",
    "content": "You are a teacher whose containers refuse to start"
  },
  {
    "role": "user",
    "content": "Let’s wrap up the class already"
  }
]
}'
```

And that is *exactly* the same as what we got via `chat.completions.create`. See the [llama.cpp server docs](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#post-v1chatcompletions-openai-compatible-chat-completions-api) for details.

---

Below is extra information that we did not cover practically during the session.

## SGLang

SGLang is launched with a server in exactly the same way ([docs](https://docs.sglang.ai/start/install.html)):

```bash
docker run --gpus all \
    -p 8000:8000 \
    -v ~/LLM/models/:/root/models \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path /root/models/Qwen2-0.5B-Instruct --host 0.0.0.0 --port 8000 --trust-remote-code
```

But amusingly, it runs out of GPU memory for this model, whereas vLLM coped without extra tricks:

```
torch.OutOfMemoryError: CUDA out of memory.
```

Trying to launch an AWQ-quantised model reveals that the Docker image lacks **vLLM** inside itself (!):

```
ValueError: awq quantization requires some operators from vllm. Please install vllm by `pip install vllm==0.7.2`
```

The fix is probably to rebuild the image, inheriting from the original and installing vLLM during build; at that point you might as well go back to vLLM directly.

Launching an **AWQ** model in **vLLM** with default settings:

```bash
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-AWQ ~/LLM/models/Qwen2-0.5B-Instruct-AWQ

sudo docker run --runtime nvidia --gpus all \
    -v ~/LLM/models/:/root/models \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --trust-remote-code \
    --model /root/models/Qwen2-0.5B-Instruct-AWQ --quantization awq_marlin
```

```
Maximum concurrency for 32,768 tokens per request: 6.59x
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

It is worth experimenting with the optimisation flags in SGLang’s docs. They also promise many *on-the-fly* quantisation schemes ([https://docs.sglang.ai/backend/quantization.html#online-quantization](https://docs.sglang.ai/backend/quantization.html#online-quantization)). Definitely keep an eye on the project; on a large GPU it may prove faster and more stable than vLLM, as claimed by the authors and some NLP researchers in chat rooms. I’ll add impressions from an A100 run (say, Qwen2.5 32B in full precision) once I get hold of one. The FP8-quantised model also failed to load because it apparently first wants to allocate the full-precision weights in memory.

## APHRODITE

```bash
docker run --runtime nvidia --gpus all \
    -v ~/LLM/models/:/root/models \
    -p 8000:2242 \
    --ipc=host \
    alpindale/aphrodite-openai:latest --trust_remote_code \
    --model /root/models/Qwen2-0.5B-Instruct 
```

This one surprised me. It starts faster and uses less GPU memory. I ran it for the first time while writing this guide; now I think it’s worth testing in production on a big GPU. As mentioned in class, it is written in C++ on top of vLLM, about 80% C++.

## TEXT GENERATION INFERENCE – TGI (by HuggingFace)

```bash
docker run --gpus all -p 8080:80 \
    -v ~/LLM/models/:/root/models \
    ghcr.io/huggingface/text-generation-inference:3.2.1 \
    --model-id /root/models/Qwen2-0.5B-Instruct 
```

All else being equal, with default settings this returned a hallucination full of Chinese characters in response to the same question we asked every other server.

## Performance Tests

```python
import asyncio
import openai
from datetime import datetime

base_url = 'http://localhost:8000/v1'
api_key = 'dummy_key'

async def send_completion_request(prompt, model="/root/models/Qwen2-0.5B-Instruct"):
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response

async def main():
    prompts = [
        "Tell a fairy tale and write Python code about it" * 200,
        "Tell a fairy tale and write Python code about it" * 200,
        "Tell a fairy tale and write Python code about it" * 200,
        "Tell a fairy tale and write Python code about it" * 200,
        "Tell a fairy tale and write Python code about it" * 200,
        "Tell a fairy tale and write Python code about it" * 200,
        "Tell a fairy tale and write Python code about it" * 200,
    ]
    
    tasks = [send_completion_request(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return responses

s = datetime.now()
r = await main()
e = datetime.now()

print(e - s)
```

Average times (the first request is usually slower):

```
Aphrodite:
0:00:01.927360

TGI:
0:00:00.623251

vLLM:
0:00:00.993214
```
