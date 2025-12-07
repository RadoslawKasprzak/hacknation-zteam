from safety_agent import safety_agent
from scenario_agent_with_verificator import scenario_agent_with_verificator
from specialized_search import specialized_agents
import json

#dane z frontu
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

# tu powinna nastąpić normalizacja wag? tj. skala od 1-100 i normalizacja do 0-1 (float)?

def extract_country_and_fields(sanitized_json):
    """
    From safety agent output, extract the main country and all field names.
    Returns a dict with:
      - country: string
      - fields: list of field names (area_name)
    """
    global_areas = sanitized_json.get("global_areas", [])
    fields = []
    country = None

    for area in global_areas:
        fields.append(area.get("area_name", "Unknown"))
        for analysis in area.get("country_impact_analysis", []):
            # take the first country mentioned
            if not country and "country" in analysis:
                # assuming format: "Atlantis: <text>"
                country = analysis["country"].split(":", 1)[0].strip()

    return {"country": country, "fields": fields}


for scenario, weight in scenarios:

    # PLLUM agent \/
    resp = scenario_agent_with_verificator(user_prompt, scenario, weight)
    countries, subjects = resp['countries'], resp['subjects']

    #  PLLUM safety agent \/
    sanitized_context_str, _ = safety_agent(user_prompt, scenario)

    # launch specialized agent using the **string** directly
    specialized_agents(sanitized_context_str)

  # external_results = []
  # # External Agents \/
  # for subject in subjects_to_check:
  #   raw_content1, summary1, url1 = web_search_agent(sanitized_scenario, sanitized_user_prompt, subject, countries_to_check, preferred_domains)
  #   raw_content2, summary2, url2 = web_search_agent(sanitized_scenario, sanitized_user_prompt, subject, countries_to_check, preferred_domains)
  #
  #   #throw error if comparison fails
  #   try:
  #     raw_content, summary, url = compare_and_verify_web_result(sanitized_scenario, sanitized_user_prompt, subject, countries_to_check, preferred_domains, raw_content1, summary1, url1, raw_content2, summary2, url2)
  #   except:
  #     continue
  #
  #   external_results.append((raw_content, summary, url, scenario, weight))


  # analiza PLLUM
  #analyse(user_prompt, scenarios, external_results)