from safety_agent import safety_agent
from scenario_agent_with_verificator import scenario_agent_with_verificator
from specialized_search import specialized_agents


# tu powinna nastąpić normalizacja wag? tj. skala od 1-100 i normalizacja do 0-1 (float)?
def run_engine(scenarios_in):

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
    """, scenarios_in)

    for s in scenarios:
      # unpack obj
      scenario, weight = (s['text'], s['weight'])

      # PLLUM agent \/
      resp = scenario_agent_with_verificator(user_prompt, scenario, weight)
      countries, subjects = (resp['countries'], resp['subjects'])

      # PLLUM agent \/
      sanitized_user_prompt, sanitized_scenario = safety_agent(user_prompt, scenario)

      # SPECIALIZED AGENTS
      # specialized_agents(sanitized_user_prompt)

    result_report = "not yet implemented "+str(sanitized_scenario)
    return result_report
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
      #
      #
      # # analiza PLLUM
      # analyse(user_prompt, scenarios, external_results)