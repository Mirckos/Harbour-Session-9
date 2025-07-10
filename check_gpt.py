import httpx
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080",
    api_key="12312312",
    http_client=httpx.Client(http2=False)
)

response = client.chat.completions.create(
    model="qwen2.5-0.5b-instruct-q4_k_m",
    messages=[{"role": "user", "content": "Hi, how are you?"}],
    max_tokens=128,
)

print(response.choices[0].message.content.strip())
