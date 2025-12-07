import json
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

import config2
from safety_agent import safety_agent
from scenario_agent_with_verificator import scenario_agent_with_verificator


class PredictiveImpactAgent:


    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=lambda: config2.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=900,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Jesteś analitykiem strategicznym fikcyjnego państwa Atlantis. "
                "Na podstawie podanego kontekstu państwa, opisu scenariusza oraz analiz wpływu "
                "działań innych krajów masz przygotować PROGNOZĘ dla Atlantis. "
                "Prognoza ma obejmować dwa horyzonty czasowe (12 miesięcy i 36 miesięcy) "
                "oraz dwa warianty: pozytywny i negatywny. "
                "Oceniaj konsekwencje dla gospodarki, bezpieczeństwa, społeczeństwa i pozycji międzynarodowej Atlantis. "
                "Nie wymyślaj faktów sprzecznych z danymi, ale możesz realistycznie EKstrapolować trendy."
            ),
            (
                "user",
                "Kontekst państwa Atlantis:\n{home_context}\n\n"
                "Scenariusz sytuacyjny:\n{scenario}\n\n"
                "Analizy wpływu zewnętrznego (dla różnych krajów i tematów):\n{external_analyses}\n\n"
                "Zadanie:\n"
                "Przygotuj PROGNOZĘ dla państwa Atlantis w następującej, DOKŁADNIE określonej strukturze:\n\n"
                "{{\n"
                '  "12m_positive": "<prognoza pozytywna na ok. 12 miesięcy>",\n'
                '  "12m_negative": "<prognoza negatywna na ok. 12 miesięcy>",\n'
                '  "36m_positive": "<prognoza pozytywna na ok. 36 miesięcy>",\n'
                '  "36m_negative": "<prognoza negatywna na ok. 36 miesięcy>"\n'
                "}}\n\n"
                "Wymogi:\n"
                "- każda z czterech prognoz powinna mieć 3–6 zdań,\n"
                "- pisz po polsku, z perspektywy mieszkańców Atlantis (co to dla nich znaczy),\n"
                "- uwzględnij możliwe zmiany w cenach, rynku pracy, bezpieczeństwie, inwestycjach, relacjach międzynarodowych,\n"
                "- jeśli to możliwe, delikatnie odwołaj się do liczb/trendów z analiz wejściowych (bez wymyślania nowych konkretnych liczb),\n"
                "- ODPOWIEDZ WYŁĄCZNIE poprawnym JSON-em, bez żadnego dodatkowego tekstu ani komentarza."
            ),
        ])

    def predict_for_scenario(
        self,
        home_context: str,
        scenario: str,
        external_results: Dict[str, Dict[str, str]],
    ) -> Dict[str, str] | None:
        """
        Zwraca słownik:
        {
          "12m_positive": "...",
          "12m_negative": "...",
          "36m_positive": "...",
          "36m_negative": "..."
        }
        albo None w razie błędu.
        """

        # Zamieniamy external_results (kraj -> temat -> analiza) na tekst
        try:
            external_analyses_text = json.dumps(
                external_results,
                ensure_ascii=False,
                indent=2
            )
        except TypeError:
            external_analyses_text = str(external_results)

        messages = self.prompt.format_messages(
            home_context=home_context,
            scenario=scenario,
            external_analyses=external_analyses_text,
        )

        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            print("❌ Błąd LLM (predykcja):", e)
            return None

        raw = response.content.strip()
        print("\n=== [DEBUG] RAW PREDICTION JSON ===")
        print(raw)

        # próba parsowania JSON-a
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print("❌ JSONDecodeError w predykcji:", e)
            return None

        # prosta walidacja kluczy
        required_keys = ["12m_positive", "12m_negative", "36m_positive", "36m_negative"]
        if not all(k in parsed for k in required_keys):
            print("❌ Brak wymaganych kluczy w predykcji:", parsed)
            return None

        return parsed
