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
    - pisze po polsku, w formacie Markdown,
    - długość raportu: ok. 2000 słów.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=lambda: config2.OPENAI_API_KEY,
            temperature=0.3,
            # 🔼 zwiększamy limit, żeby zmieścić ok. 2000 słów
            max_tokens=4000,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Jesteś głównym analitykiem strategicznym państwa Atlantis. "
                "Masz z wielu agentów cząstkowe PREDYKCJE, każda dla innego scenariusza. "
                "Twoim zadaniem jest napisać JEDEN zbiorczy raport dla rządu Atlantis. "
                "Raport ma być po polsku, klarowny, zrozumiały dla decydentów. "
                "Raport powinien mieć około 2000 słów (nie mniej niż 1800 i nie więcej niż 2200 słów)."
            ),
            (
                "user",
                "Kontekst państwa Atlantis:\n{home_context}\n\n"
                "Dane wejściowe (lista scenariuszy z predykcjami):\n{scenarios_json}\n\n"
                "Zadanie:\n"
                "Na bazie powyższych danych przygotuj raport końcowy w formacie MARKDOWN.\n"
                "Struktura raportu (NIE dodawaj innych głównych nagłówków):\n\n"
                "# Raport strategiczny dla państwa Atlantis\n"
                "## Horyzont 12 miesięcy\n"
                "### Scenariusze – ujęcie pozytywne\n"
                "- szczegółowo podsumuj, jakie pozytywne ścieżki 12-miesięczne pojawiają się w różnych scenariuszach,\n"
                "- wskaż wspólne elementy (np. gdzie scenariusze są zgodne),\n"
                "- opisz mechanizmy (jak konkretne zjawiska prowadzą do tych pozytywnych efektów),\n"
                "- wskaż 2–3 kluczowe szanse i rozwiń je na 2–3 zdania każda.\n\n"
                "### Scenariusze – ujęcie negatywne\n"
                "- szczegółowo podsumuj główne zagrożenia w horyzoncie 12 miesięcy,\n"
                "- wskaż obszary największego ryzyka (gospodarka, bezpieczeństwo, społeczeństwo, pozycja międzynarodowa),\n"
                "- opisz możliwe łańcuchy zdarzeń (jak dane ryzyko może się rozwinąć),\n"
                "- wskaż 2–3 najważniejsze punkty, które rząd powinien monitorować i do każdego dodaj 2–3 zdania wyjaśnienia.\n\n"
                "## Horyzont 36 miesięcy\n"
                "### Scenariusze – ujęcie pozytywne\n"
                "- podsumuj długoterminowe szanse w różnych scenariuszach,\n"
                "- wskaż, jakie inwestycje/opcje strategiczne są powtarzalne w wielu scenariuszach,\n"
                "- opisz, jak te szanse mogą zmienić strukturę gospodarki, bezpieczeństwa i pozycji międzynarodowej,\n"
                "- dodaj 2–3 krótkie przykłady możliwych pozytywnych ścieżek rozwoju.\n\n"
                "### Scenariusze – ujęcie negatywne\n"
                "- opisz możliwe długoterminowe ryzyka, jeśli rzeczy pójdą źle,\n"
                "- podkreśl, jakie konsekwencje mogą być trwałe i trudne do odwrócenia,\n"
                "- wskaż, które scenariusze są najbardziej niebezpieczne dla stabilności państwa i społeczeństwa,\n"
                "- rozwiń 2–3 potencjalne \"czarne scenariusze\" w kilku zdaniach każdy.\n\n"
                "## Rekomendacje dla rządu Atlantis\n"
                "- wypisz 5–7 konkretnych rekomendacji (lista punktowana),\n"
                "- każda rekomendacja maksymalnie 2–3 zdania,\n"
                "- rekomendacje mają wynikać z tego, co widzisz w predykcjach i powtarzających się motywach,\n"
                "- wskaż, które rekomendacje są kluczowe w krótkim (12m), a które w długim (36m) horyzoncie.\n\n"
                "WAŻNE:\n"
                "- Odnoś się do scenariuszy ogólnie (np. \"w części scenariuszy zakłada się...\", \"w scenariuszach z silnym kryzysem...\").\n"
                "- NIE cytuj całych predykcji, tylko je streszczaj i syntetyzuj.\n"
                "- Pisz spójnie – raport ma być czytany jak jedno opracowanie, a nie zlepek notatek.\n"
                "- Pilnuj długości: ok. 2000 słów (1800–2200). Jeśli trzeba, rozbuduj argumentację i przykłady.\n"
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
