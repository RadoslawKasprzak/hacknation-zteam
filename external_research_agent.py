import json
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

import config2
from safety_agent import safety_agent
from scenario_agent_with_verificator import scenario_agent_with_verificator


# ===================== KLASA: EXTERNAL RESEARCH AGENT =====================

class ExternalResearchAgent:
    """
    Agent do zewnętrznego researchu:
    - korzysta z Tavily (web search),
    - analizuje wyniki GPT-4.1,
    - zwraca krótkie (~6 zdań) analizy dla par (kraj, temat).
    """

    def __init__(self,
                 max_results: int = 5,
                 search_depth: str = "advanced"):
        # LLM (GPT-4.1)
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=lambda: config2.OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=800,
        )

        # Tavily (web search)
        self.search_tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
        )

        # Prompt do analizy (~6 zdań)
        self.research_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Jesteś analitykiem Ministerstwa Spraw Zagranicznych. "
                "Na podstawie dostarczonych wyników wyszukiwania przygotuj "
                "krótką, aktualną analizę sytuacji w danym kraju w kontekście podanego tematu. "
                "Odnoś się WYŁĄCZNIE do informacji zawartych w wynikach wyszukiwania. "
                "Nie dodawaj żadnych domysłów spoza danych."
            ),
            (
                "user",
                "Kraj: {country}\n"
                "Temat: {subject}\n\n"
                "Wyniki wyszukiwania (surowe dane):\n{search_results}\n\n"
                "Zadanie:\n"
                "- napisz dokładnie OKOŁO 6 ZDAŃ (nie mniej niż 5, nie więcej niż 7),\n"
                "- skup się na NAJNOWSZYCH wydarzeniach (ostatnie miesiące),\n"
                "- uwzględnij decyzje rządowe, inwestycje, embarga, kryzysy lub reformy,\n"
                "- wskaż 1–2 główne ryzyka,\n"
                "- styl: analityczny, zwięzły, po polsku."
            ),
        ])

    def research_country_subject(self, country: str, subject: str) -> str:
        """
        Dla danego kraju i tematu:
          1) pyta Tavily o najnowsze info,
          2) wrzuca wyniki do GPT-4.1,
          3) zwraca analizę ~6 zdań po polsku.
        """

        query = (
            f"najnowsze informacje o temacie '{subject}' w kraju {country}, "
            f"lata 2024-2025, polityka, gospodarka, bezpieczeństwo, decyzje rządowe"
        )

        # --- krok 1: Tavily search ---
        try:
            search_results = self.search_tool.invoke({"query": query})
        except Exception as e:
            print(f"❌ Błąd Tavily dla {country} / {subject}: {e}")
            return "Nie udało się pobrać danych z wyszukiwarki."

        # (opcjonalnie) debug surowych wyników
        print(f"\n=== [DEBUG] Tavily raw results for {country} / {subject} ===")
        try:
            print(json.dumps(search_results, indent=2, ensure_ascii=False))
        except TypeError:
            print(search_results)

        # --- krok 2: budowanie promptu dla LLM ---
        messages = self.research_prompt.format_messages(
            country=country,
            subject=subject,
            search_results=search_results,
        )

        # --- krok 3: wywołanie GPT-4.1 ---
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"❌ Błąd LLM dla {country} / {subject}: {e}")
            return "Nie udało się wygenerować analizy."

    def research_matrix(self,
                        countries: List[str],
                        subjects: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Dla listy krajów i tematów zwraca słownik:
        {
          kraj: {
            temat: "analiza ~6 zdań",
            ...
          },
          ...
        }
        """
        results: Dict[str, Dict[str, str]] = {}

        for country in countries:
            results[country] = {}
            for subject in subjects:
                print("\n" + "#" * 80)
                print(f"### KRAJ: {country} | TEMAT: {subject}")
                print("#" * 80)

                summary = self.research_country_subject(country, subject)
                results[country][subject] = summary

                print("\n--- ANALIZA (~6 zdań) ---")
                print(summary)

        return results


# ===================== DANE Z FRONTU (ATLANTIS + SCENARIUSZE) =====================

user_prompt, scenarios = ("""
Nazwa państwa: Atlantis

Istotne cechy położenia geograficznego: dostęp do Morza Bałtyckiego, kilka dużych
żeglownych rzek, ograniczone zasoby wody pitnej

Liczba ludności: 28 mln

Budzet Wojskowy: 11 mld euro  ==> POUFNE

Klimat: umiarkowany

Silne strony gospodarki: przemysł ciężki, motoryzacyjny, spożywczy, chemiczny, ICT, ambicje
odgrywania istotnej roli w zakresie OZE, przetwarzania surowców krytycznych oraz budowy
ponadnarodowej infrastruktury AI (m.in. big data centers, giga fabryki AI, komputery
kwantowe)

Liczebność armii: 150 tys. zawodowych żołnierzy

Stopnień cyfryzacji społeczeństwa: powyżej średniej europejskiej

Waluta: inna niż euro

Kluczowe relacje dwustronne: Niemcy, Francja, Finlandia, Ukraina, USA, Japonia
Potencjalne zagrożenia polityczne i gospodarcze: niestabilność w UE, rozpad UE na grupy
„różnych prędkości” pod względem tempa rozwoju oraz zainteresowania głębszą integracją;
negatywna kampania wizerunkowa ze strony kilku aktorów państwowych wymierzona przeciw
rządowi lub społeczeństwu Atlantis; zakłócenia w dostawach paliw węglowodorowych z USA,
Skandynawii, Zatoki Perskiej (wynikające z potencjalnych zmian w polityce wewnętrznej
krajów eksporterów lub problemów w transporcie, np. ataki Hutich na gazowce na Morzu
Czerwonym); narażenie na spowolnienie rozwoju sektora ICT z powodu embarga na
wysokozaawansowane procesory

Potencjalne zagrożenie militarne: zagrożenie atakiem zbrojnym jednego
z sąsiadów; trwające od wielu lat ataki hybrydowe co najmniej jednego sąsiada, w tym
w obszarze infrastruktury krytycznej i cyberprzestrzeni

Kamienie milowe w rozwoju politycznym i gospodarczym: demokracja parlamentarna od 130
lat; okres stagnacji gospodarczej w latach 1930-1950 oraz 1980-1990; członkostwo w UE i
NATO od roku 1997; 25. gospodarka świata wg PKB od roku 2020; deficyt budżetowy oraz
dług publiczny w okolicach średniej unijnej
""",
[("Wskutek zaistniałej przed miesiącem katastrofy naturalnej wiodący światowy "
  "producent procesorów graficznych stracił 60% zdolności produkcyjnych; odbudowa "
  "mocy produkcyjnych poprzez inwestycje w filie zlokalizowane na obszarach nieobjętych "
  "katastrofą potrwa do końca roku 2028", 30),
 ("Przemysł motoryzacyjny w Europie (piątka głównych partnerów handlowych państwa Atlantis"
  " to kraje europejskie) bardzo wolno przestawia się na produkcję samochodów elektrycznych; "
  "rynek europejski zalewają tanie samochody elektryczne z Azji Wschodniej; europejski przemysł "
  "motoryzacyjny będzie miał w roku 2025 zyski na poziomie 30% średnich rocznych zysków z lat 2020-2024", 15)])


# ===================== GŁÓWNA PĘTLA SCENARIUSZY =====================

if __name__ == "__main__":
    external_agent = ExternalResearchAgent()
    all_external_results_per_scenario = []

    for s in scenarios:
        # unpack obj
        scenario, weight = s

        print("\n" + "=" * 100)
        print(f"SCENARIUSZ (waga={weight}):")
        print(scenario)
        print("=" * 100)

        # PLLUM agent – wybór krajów i tematów
        resp = scenario_agent_with_verificator(user_prompt, scenario, weight)
        if isinstance(resp, dict) and 'countries' in resp and 'subjects' in resp:
            countries = resp['countries']
            subjects = resp['subjects']
        else:
            print("❌ BŁĘDNA STRUKTURA DANYCH Z PLLUM:", resp)
            countries = []
            subjects = []

        # Safety agent – usunięcie poufnych danych
        sanitized_user_prompt, sanitized_scenario = safety_agent(user_prompt, scenario)

        print("\n===== OCZYSZCZONY SCENARIUSZ =====")
        print(sanitized_scenario)

        # External research – tylko jeśli mamy kraje i tematy
        if countries and subjects:
            external_results = external_agent.research_matrix(countries, subjects)
        else:
            external_results = {}

        # Zapisz wyniki dla tego scenariusza (opcjonalnie)
        all_external_results_per_scenario.append({
            "scenario": scenario,
            "weight": weight,
            "countries": countries,
            "subjects": subjects,
            "external_results": external_results,
        })

    # (opcjonalnie) zapis do pliku
    with open("external_results.json", "w", encoding="utf-8") as f:
        json.dump(all_external_results_per_scenario, f, ensure_ascii=False, indent=2)

    print("\n✅ Zakończono analizę, wyniki zapisane do external_results.json")
