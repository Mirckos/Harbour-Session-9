from openai import OpenAI, AsyncOpenAI #async as you want
from datetime import datetime

api_base = "http://127.0.0.1:8005"
api_key = "dummy_key"

client_emb = OpenAI(api_key=api_key ,
                    base_url=api_base)

models_list = [client_emb.models.list().data[i].id for i in range(len(client_emb.models.list().data))]

print(datetime.now())

for model in models_list:

    responses = client_emb.embeddings.create(    input=[
        "Hello everyone, we’re from the Central University",
        "This week we’re deploying various large language models",
    ],
                                             model=model)
    for data in responses.data:
        print(data.embedding[:5])
        print(len(data.embedding))
        print(model)

print(datetime.now())
