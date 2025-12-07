import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config2


class SummaryReportAgent:
    """
    Agent, który:
    - bierze listę scenariuszy z predykcjami,
    - łączy wszystko w jeden raport,
    - dzieli na: 12m/36m oraz pozytywny/negatywny,
    - pisze po polsku, w formacie Markdown.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=lambda: config2.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=2000,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Jesteś głównym analitykiem strategicznym państwa Atlantis. "
                "Masz z wielu agentów cząstkowe PREDYKCJE, każda dla innego scenariusza. "
                "Twoim zadaniem jest napisać JEDEN zbiorczy raport dla rządu Atlantis. "
                "Raport ma być po polsku, klarowny, zrozumiały dla decydentów."
            ),
            (
                "user",
                "Kontekst państwa Atlantis:\n{home_context}\n\n"
                "Dane wejściowe (lista scenariuszy z predykcjami):\n{scenarios_json}\n\n"
                "Zadanie:\n"
                "Na bazie powyższych danych przygotuj raport końcowy w formacie MARKDOWN.\n"
                "Struktura raportu:\n\n"
                "# Raport strategiczny dla państwa Atlantis\n"
                "## Horyzont 12 miesięcy\n"
                "### Scenariusze – ujęcie pozytywne\n"
                "- podsumuj, jakie pozytywne ścieżki 12-miesięczne pojawiają się w różnych scenariuszach,\n"
                "- wskaż wspólne elementy (np. gdzie scenariusze są zgodne),\n"
                "- wskaż 2–3 kluczowe szanse.\n\n"
                "### Scenariusze – ujęcie negatywne\n"
                "- podsumuj główne zagrożenia w horyzoncie 12 miesięcy,\n"
                "- wskaż obszary największego ryzyka (gospodarka, bezpieczeństwo, społeczeństwo, pozycja międzynarodowa),\n"
                "- wskaż 2–3 najważniejsze punkty, które rząd powinien monitorować.\n\n"
                "## Horyzont 36 miesięcy\n"
                "### Scenariusze – ujęcie pozytywne\n"
                "- podsumuj długoterminowe szanse w różnych scenariuszach,\n"
                "- wskaż, jakie inwestycje/opcje strategiczne są powtarzalne w wielu scenariuszach.\n\n"
                "### Scenariusze – ujęcie negatywne\n"
                "- opisz możliwe długoterminowe ryzyka, jeśli rzeczy pójdą źle,\n"
                "- podkreśl, jakie konsekwencje mogą być trwałe i trudne do odwrócenia.\n\n"
                "## Rekomendacje dla rządu Atlantis\n"
                "- wypisz 5–7 konkretnych rekomendacji (krótkie, w formie listy punktowanej),\n"
                "- każda rekomendacja ma być maksymalnie 1–2 zdania,\n"
                "- rekomendacje mają wynikać z tego, co widzisz w predykcjach.\n\n"
                "WAŻNE:\n"
                "- Odnoś się do scenariuszy ogólnie (np. \"w części scenariuszy zakłada się...\", \"w scenariuszach z silnym kryzysem...\"),\n"
                "- NIE cytuj całych predykcji, tylko je streszczaj,\n"
                "- NIE dodawaj żadnych nagłówków poza wskazanymi powyżej,\n"
                "- Pisz w sposób zrozumiały, ale merytoryczny.\n"
            ),
        ])

    def build_global_report(self, home_context: str, scenarios_data: list[dict]) -> str:

        compact = []
        for item in scenarios_data:
            compact.append({
                "scenario": item.get("scenario"),
                "weight": item.get("weight"),
                "predictions": item.get("predictions"),
            })

        scenarios_json = json.dumps(compact, ensure_ascii=False, indent=2)

        messages = self.prompt.format_messages(
            home_context=home_context,
            scenarios_json=scenarios_json,
        )

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print("❌ Błąd LLM (raport zbiorczy):", e)
            return "# Raport strategiczny dla państwa Atlantis\n\nNie udało się wygenerować raportu."
