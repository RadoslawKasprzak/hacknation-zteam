import json
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

import config2
from safety_agent import safety_agent
from scenario_agent_with_verificator import scenario_agent_with_verificator


# ===================== KLASA: EXTERNAL RESEARCH AGENT =====================
class PredictiveImpactAgent:
    """
    Agent predykcyjny:
    - bierze: kontekst Atlantis, scenariusz, analizy z external agenta,
    - zwraca prognozy na 12 i 36 miesięcy,
    - w dwóch wariantach: pozytywnym i negatywnym,
    - wyłącznie dla państwa Atlantis.
    """

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

        # ✅ PROMPT Z WYMUSZENIEM LICZB
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
                "- ✅ jeśli w danych występują JAKIEKOLWIEK LICZBY (kwoty, %, MW, eksport, import), "
                "MUSISZ przytoczyć co najmniej 1–2 takie liczby,\n"
                "- ✅ NIE WOLNO wymyślać liczb,\n"
                "- ✅ jeśli w źródłach NIE MA LICZB, musisz jasno napisać: "
                "„w dostępnych źródłach nie podano konkretnych danych liczbowych”,\n"
                "- pisz po polsku, prostym językiem."
            ),
        ])

    # ===================== ANALIZA JEDNEGO KRAJU =====================

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

        # --- Tavily ---
        try:
            search_results = self.search_tool.invoke({"query": query})
        except Exception as e:
            print(f"❌ Tavily error: {e}")
            return "Nie udało się pobrać danych z internetu."

        print(f"\n=== [DEBUG] TAVILY: {foreign_country} | {subject} ===")
        try:
            print(json.dumps(search_results, indent=2, ensure_ascii=False))
        except:
            print(search_results)

        # --- GPT ---
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

    # ===================== ANALIZA WSZYSTKICH KRAJÓW (Z PAUZĄ ENTER) =====================

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
            print("⏳ Po zakończeniu naciśnij ENTER, aby przejść dalej")
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

            # ✅ PAUZA
            input(f"\n✅ Zakończono analizę dla kraju {country}. Naciśnij ENTER, aby kontynuować...")

        return results


# ===================== DANE Z FRONTU =====================


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

# ===================== GŁÓWNA PĘTLA =====================

if __name__ == "__main__":

    external_agent = ExternalResearchAgent()
    predictive_agent = PredictiveImpactAgent()  # 👈 NOWOŚĆ
    all_external_results_per_scenario = []

    HOME_COUNTRY_NAME = "Atlantis"

    for scenario, weight in scenarios:

        print("\n" + "=" * 100)
        print(f"SCENARIUSZ (waga={weight}):")
        print(scenario)
        print("=" * 100)

        # PLLUM – kraje + tematy
        resp = scenario_agent_with_verificator(user_prompt, scenario, weight)

        if isinstance(resp, dict) and "countries" in resp and "subjects" in resp:
            countries = resp["countries"][:1]  # ✅ tylko pierwsze państwo
            subjects = resp["subjects"][:1]
        else:
            print("❌ BŁĘDNA STRUKTURA:", resp)
            continue

        # Safety Agent
        sanitized_user_prompt, sanitized_scenario = safety_agent(user_prompt, scenario)

        print("\n===== OCZYSZCZONY SCENARIUSZ =====")
        print(sanitized_scenario)

        # External Analysis
        external_results = external_agent.analyze_matrix_for_scenario(
            home_country_name=HOME_COUNTRY_NAME,
            home_context=sanitized_user_prompt,
            scenario=sanitized_scenario,
            foreign_countries=countries,
            subjects=subjects,
        )
        predictions = predictive_agent.predict_for_scenario(
            home_context=sanitized_user_prompt,
            scenario=sanitized_scenario,
            external_results=external_results,
        )

        print("\n===== PREDYKCJA DLA ATLANTIS (12 / 36 miesięcy) =====")
        print(json.dumps(predictions, ensure_ascii=False, indent=2))

        all_external_results_per_scenario.append({
            "scenario": scenario,
            "weight": weight,
            "countries": countries,
            "subjects": subjects,
            "external_results": external_results,
            "predictions": predictions,  # 👈 NOWOŚĆ
        })

    # ✅ ZAPIS DO PLIKU
    with open("external_results.json", "w", encoding="utf-8") as f:
        json.dump(all_external_results_per_scenario, f, ensure_ascii=False, indent=2)

    print("\n✅ GOTOWE – wszystkie analizy zapisane do external_results.json")
