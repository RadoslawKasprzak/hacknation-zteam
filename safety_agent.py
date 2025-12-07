import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config2


# ✅ POPRAWIONY PROMPT SYSTEMOWY (literówki + jednoznaczność)
system_prompt = """Jesteś analitykiem bezpieczeństwa dla fikcyjnego kraju Atlantis.
W tekście, który dostaniesz, usuń TYLKO dane poufne.
Zwróć WYŁĄCZNIE poprawny obiekt JSON z DWOMA polami: scenario oraz context.
Nie dodawaj żadnego dodatkowego tekstu.
"""

user_prompt_template = """scenario:
{scenario}

context:
{context}
"""

api_key = config2.OPENAI_API_KEY
model_name = "gpt-4.1"

# ✅ STABILNA KONFIGURACJA LLM
llm = ChatOpenAI(
    model=model_name,
    api_key=lambda: api_key,
    temperature=0.4,   # mniej losowości = stabilniejszy JSON
    max_tokens=8000   # rozsądny limit
)


def load_json(path):
    """Wczytuje plik JSON z dysku"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safety_agent(user_prompt, scenario):
    """Usuwa dane poufne z treści przy użyciu AI"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt_template)
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "scenario": scenario,
            "context": user_prompt
        })

        # ✅ DEBUG - surowa odpowiedź modelu
        print("✅ SUROWA ODPOWIEDŹ LLM:")
        print(response.content)

        # ✅ PRÓBA PARSOWANIA JSON
        loads = json.loads(response.content)

        # ✅ WALIDACJA KLUCZY
        if "context" not in loads or "scenario" not in loads:
            raise ValueError("Brakuje kluczy 'context' lub 'scenario' w odpowiedzi JSON")

        # ✅ ZWRACANY BEZPIECZNY TEKST
        print("✅ OCZYSZCZONY CONTEXT:")
        print(loads["context"])

        print("✅ OCZYSZCZONY SCENARIO:")
        print(loads["scenario"])

        return loads["context"], loads["scenario"]

    except json.JSONDecodeError:
        print("❌ BŁĄD: Model nie zwrócił poprawnego JSON-a!")
        print(response.content)
        return None, None

    except Exception as e:
        print("❌ BŁĄD KRYTYCZNY:")
        print(e)
        return None, None
