import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

load_dotenv()
async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)


class BeeAnswer(BaseModel):
    answer: str
    confidence: float
    sources: list[str]
    vet_needed: bool


def get_hive_status(hive_id: int):
    # Заглушка
    return {
        "hive_id": hive_id,
        "queen_present": True,
        "brood_frames": 6,
        "status": "healthy",
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_hive_status",
            "description": "Получить статус улья по его номеру",
            "parameters": {
                "type": "object",
                "properties": {"hive_id": {"type": "integer"}},
                "required": ["hive_id"],
            },
        },
    }
]


async def bee_assistant(question):
    system_promt = 'Ты помощник пчеловода. Стиль ответов формальный. Любая тема кроме пчеловодства должна пресекаться. Ты должен отвечать "Я помогаю только по темам связаным с пчеловодством" Отвечай СТРОГО в формате JSON с полями: answer (str), confidence (0-1) (float), sources (list[str]), vet_needed (bool) (Нужен ли специалист для более качественного ответа на вопрос). Никакого текста вне JSON.'
    messages = [
        {"role": "system", "content": system_promt},
        {"role": "user", "content": question},
    ]
    resp = await async_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = get_hive_status(**args)
            messages.append(msg)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
            final = await async_client.chat.completions.create(
                model=MODEL, messages=messages
            )

            raw = final.choices[0].message.content
            data = json.loads(raw)
            return BeeAnswer(**data)
    else:
        raw = msg.content
        data = json.loads(raw)
        return BeeAnswer(**data)


async def main():
    while True:
        user_question = input(
            'Введите свой вопрос на тему пчеловодства.\nВы можете запросить у нейросети информацию о своих ульях по номеру улья.\nЧтобы закончить введите "0":\n'
        )
        if user_question == "0":
            break
        resp = await bee_assistant(user_question)
        print(resp.answer)
        print("\nИсточники:")
        if len(resp.sources) != 0:
            for source in resp.sources:
                print(f"\n- {source}")
        print(f"\n\nУверенность в ответе: {resp.confidence}")
        if resp.vet_needed:
            print(
                "\nВам стоит дополнительно обратиться с этим вопросом к специалисту по мнению нейросети\n"
            )
        else:
            print(
                "\nНейросеть считает что дала вам достаточно ёмкий ответ чтобы вы могли не обращаться к спецалисту\n"
            )


if __name__ == "__main__":
    asyncio.run(main())
