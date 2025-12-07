import json
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

import config2
from safety_agent import safety_agent
from scenario_agent_with_verificator import scenario_agent_with_verificator

class ExternalResearchAgent:
    """
    Agent do zewnętrznego researchu:
    - Tavily (internet),
    - GPT-4.1 (analiza),
    - perspektywa mieszkańca Atlantis,
    - WYMUSZONE WIARYGODNE LICZBY.
    """

    def __init__(self, max_results: int = 5, search_depth: str = "advanced"):

        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=lambda: config2.OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=900,
        )

        # Tavily
        self.search_tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
        )

        # PROMPT Z WYMUSZENIEM LICZB
        self.research_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Tłumaczysz sytuację geopolityczną zwykłemu mieszkańcowi fikcyjnego państwa Atlantis. "
                "Wyjaśniasz, jak wydarzenia w innych krajach wpływają na jego życie: ceny, pracę, "
                "bezpieczeństwo i stabilność państwa. "
                "NIE używaj urzędniczego języka. Pisz prosto i konkretnie. "
                "WOLNO korzystać TYLKO z danych zawartych w wynikach wyszukiwania i scenariuszu. "
                "NIE WOLNO wymyślać żadnych faktów ani liczb."
            ),
            (
                "user",
                "Państwo, którego jestem obywatelem: {home_country_name}\n\n"
                "Kontekst mojego państwa (Atlantis):\n{home_context}\n\n"
                "Kraj, w którym dzieje się ważna sytuacja: {foreign_country}\n"
                "Temat: {subject}\n\n"
                "Scenariusz sytuacyjny:\n{scenario}\n\n"
                "Wyniki wyszukiwania (surowe dane):\n{search_results}\n\n"
                "Zadanie:\n"
                "- napisz OKOŁO 6 ZDAŃ (5–7 zdań),\n"
                "- co najmniej 4 zdania mają dotyczyć wpływu na życie mieszkańców Atlantis,\n"
                "- maksymalnie 1 zdanie może opisywać sam kraj {foreign_country},\n"
                "- wskaż 1–2 największe zagrożenia lub szanse,\n"
                "- jeśli w danych występują JAKIEKOLWIEK LICZBY (kwoty, %, MW, eksport, import), "
                "MUSISZ przytoczyć co najmniej 1–2 takie liczby,\n"
                "- NIE WOLNO wymyślać liczb,\n"
                "- jeśli w źródłach NIE MA LICZB, musisz jasno napisać: "
                "„w dostępnych źródłach nie podano konkretnych danych liczbowych”,\n"
                "- pisz po polsku, prostym językiem."
            ),
        ])

    def analyze_impact(
        self,
        home_country_name: str,
        home_context: str,
        foreign_country: str,
        subject: str,
        scenario: str,
    ) -> str:

        query = (
            f"najnowsze informacje o temacie '{subject}' w kraju {foreign_country}, "
            f"lata 2024-2025, gospodarka, bezpieczeństwo, handel, polityka"
        )

        # Tavily
        try:
            search_results = self.search_tool.invoke({"query": query})
        except Exception as e:
            print(f"❌ Tavily error: {e}")
            return "Nie udało się pobrać danych z internetu."

        print(f"\n=== [DEBUG] TAVILY: {foreign_country} | {subject} ===")
        try:
            print(json.dumps(search_results, indent=2, ensure_ascii=False))
        except Exception:
            print(search_results)

        # GPT
        messages = self.research_prompt.format_messages(
            home_country_name=home_country_name,
            home_context=home_context,
            foreign_country=foreign_country,
            subject=subject,
            scenario=scenario,
            search_results=search_results,
        )

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return "Nie udało się wygenerować analizy."

    def analyze_matrix_for_scenario(
        self,
        home_country_name: str,
        home_context: str,
        scenario: str,
        foreign_countries: List[str],
        subjects: List[str],
    ) -> Dict[str, Dict[str, str]]:

        results: Dict[str, Dict[str, str]] = {}

        for country in foreign_countries:
            results[country] = {}

            print("\n" + "=" * 100)
            print(f"🌍 ANALIZA DLA KRAJU: {country}")
            print("=" * 100)

            for subject in subjects:
                print("\n" + "#" * 80)
                print(f"### TEMAT: {subject}")
                print("#" * 80)

                summary = self.analyze_impact(
                    home_country_name=home_country_name,
                    home_context=home_context,
                    foreign_country=country,
                    subject=subject,
                    scenario=scenario,
                )

                results[country][subject] = summary

                print("\n--- ANALIZA (~6 zdań, z liczbami jeśli są) ---")
                print(summary)

        return results