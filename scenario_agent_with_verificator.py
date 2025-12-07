import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config2


def scenario_agent_with_verificator(user_prompt, scenario, weight):
    # Inicjalizacja klienta LLM
    llm = ChatOpenAI(
        model="gpt-4.1",
        api_key=lambda: config2.OPENAI_API_KEY,
        temperature=0.2,   # mniejsza losowość = stabilniejszy JSON
        max_tokens=8000
    )

    # Przygotowanie prostego promptu: stałe role system i user
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Jesteś analitykiem w biurze Ministerstwa Spraw Zagranicznych, "
         "określ 5 krajów (lub unii krajów) oraz 5 tematów (np. "
         "'Gospodarka naftowa' lub 'Technologie mobilne') które są istotne dla podanego "
         "kontekstu omawianego kraju, oraz opisu sytuacyjnego. "
         "Twoja odpowiedź powinna zawierać TYLKO i wyłącznie obiekt JSON z dwoma polami: "
         "countries (lista krajów), subjects (lista tematów). "
         "Nie dodawaj żadnego dodatkowego tekstu, komentarzy ani formatowania markdown."),
        ("user", "Kontekst kraju:\n{kontekst}\n\nOpis sytuacyjny:\n{opis_sytuacyjny}")
    ])

    # Połączenie promptu z modelem i wywołanie
    chain = prompt | llm

    try:
        result = chain.invoke({
            "kontekst": user_prompt,
            "opis_sytuacyjny": scenario
        })

    except Exception as e:
        print("❌ Błąd wywołania LLM (API / sieć):")
        print(e)
        return None

    # --- SUROWA ODPOWIEDŹ ---
    # result to obiekt LangChain, interesuje nas result.content
    raw = getattr(result, "content", result)

    # print("✅ Surowa odpowiedź LLM (repr):")
    # print(repr(raw))

    # Upewniamy się, że mamy string
    if not isinstance(raw, str):
        raw = str(raw)

    raw = raw.strip()

    # --- USUWANIE ```json ... ``` JEŚLI MODEL TAK ODPOWIE ---
    if raw.startswith("```"):
        parts = raw.split("```")
        # np. ```json\n{...}\n``` → środek to parts[1] lub [2] w zależności od modelu
        for part in parts:
            part = part.strip()
            if part.startswith("{") or part.startswith("["):
                raw = part
                break

    # Opcjonalnie: ucinamy wszystko przed pierwszym '{' jeśli model dodał tekst
    first_brace = raw.find("{")
    if first_brace > 0:
        raw = raw[first_brace:]

    try:
        parsed_json = json.loads(raw)
    except json.JSONDecodeError as e:
        print("❌ JSONDecodeError – model nie zwrócił poprawnego JSON-a:")
        # print("Błąd:", e)
        # print("Surowy tekst po oczyszczeniu (repr):")
        # print(repr(raw))
        # Zwracamy None, żeby reszta kodu mogła to obsłużyć i iść dalej
        return None

    # print("✅ Sparsowany wynik jako obiekt Pythona:")
    # print(parsed_json)

    return parsed_json
