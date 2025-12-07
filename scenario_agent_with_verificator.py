import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config2


def scenario_agent_with_verificator(user_prompt, scenario, weight):

    # Inicjalizacja klienta LLM - korzystamy ze wzorca stosowanego w projekcie
    llm = ChatOpenAI(model="gpt-4.1", api_key=lambda: config2.OPENAI_API_KEY)

    # Przygotowanie prostego promptu: stałe role system i user
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Jesteś analitykiem w biurze Ministerstwa Spraw Zagranicznych, "
                   "określ 5 krajów (lub unii krajów) oraz 5 tematów (np. "
                   "'Gospodarka naftowa' lub 'Technologie mobilne') które są istotne dla podanego "
                   "kontekstu omawianego kraju, oraz opisu sytuacyjnego. "
                   "Twoja odpowiedź powinna zawierać TYLKO i wyłącznie obiekt JSON z dwoma polami: countries (lista krajów), subjects (list a tematów)."),
        ("user", "Kontekst kraju:\n{kontekst}\n\nOpis sytuacyjny:\n{opis_sytuacyjny}")
    ])

    # Połączenie promptu z modelem i wywołanie
    chain = prompt | llm

    try:
        # invoke powinien zwrócić wynik; zwracamy reprezentację tekstową
        result = chain.invoke({"kontekst": user_prompt, "opis_sytuacyjny": scenario})

        print(result)
        return json.loads(result.content)

    except Exception as e:
        return f"Error invoking LLM: {e}"

