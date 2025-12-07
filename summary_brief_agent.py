from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


import config2


class SummaryBriefAgent:
    """
    Agent, który robi krótkie streszczenie (250–300 słów)
    na podstawie pełnego raportu tekstowego.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=lambda: config2.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=600,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Jesteś ekspertem od syntetyzowania informacji dla decydentów. "
                "Twoim zadaniem jest przygotować zwięzłe streszczenie raportu strategicznego, "
                "tak aby ktoś zajęty miał ogólny obraz sytuacji i kierunkowe wnioski."
            ),
            (
                "user",
                "Pełny raport (Markdown):\n{full_report}\n\n"
                "Zadanie:\n"
                "- przygotuj streszczenie o długości ok. 250–300 słów po polsku,\n"
                "- ma być JEDEN spójny tekst (bez nagłówków),\n"
                "- najpierw ogólny obraz sytuacji Atlantis,\n"
                "- potem kluczowe zagrożenia i szanse,\n"
                "- na końcu 2–3 zdania o tym, co rząd powinien zrobić w pierwszej kolejności,\n"
                "- styl: prosty, zrozumiały, ale merytoryczny,\n"
                "- NIE cytuj całych fragmentów raportu, tylko parafrazuj.\n"
            ),
        ])

    def build_brief_summary(self, full_report: str) -> str:
        messages = self.prompt.format_messages(full_report=full_report)

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print("❌ Błąd LLM (krótkie streszczenie):", e)
            return "Nie udało się wygenerować krótkiego streszczenia raportu."

