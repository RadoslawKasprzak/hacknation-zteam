from agents.scenario_agent import ScenarioAgent

MSZ_OFFICIAL_SOURCES = [
  # Niemcy
  "auswaertiges-amt.de", "bmvg.de", "bmi.bund.de", "bmwk.de",
  "bmz.de", "bmuv.de", "bmbf.de", "bmdv.bund.de", "bundesregierung.de",

  # Francja
  "diplomatie.gouv.fr", "defense.gouv.fr", "interieur.gouv.fr",
  "economie.gouv.fr", "douane.gouv.fr", "ecologie.gouv.fr",
  "enseignementsup-recherche.gouv.fr", "numerique.gouv.fr", "education.gouv.fr",

  # Wielka Brytania
  "gov.uk",

  # Rosja
  "mid.ru/en", "mil.ru", "government.ru/en", "economy.gov.ru",
  "minenergo.gov.ru", "minobrnauki.gov.ru", "edu.gov.ru", "digital.gov.ru",

  # Chiny
  "fmprc.gov.cn/mfa_eng", "eng.mod.gov.cn", "gov.cn", "en.ndrc.gov.cn",
  "english.mee.gov.cn", "moe.gov.cn", "english.miit.gov.cn",

  # Indie
  "mea.gov.in", "mod.gov.in", "mha.gov.in", "commerce.gov.in",
  "powermin.gov.in", "moef.gov.in", "education.gov.in",
  "digitalindia.gov.in", "dst.gov.in",

  # Arabia Saudyjska
  "mofa.gov.sa", "mod.gov.sa", "moi.gov.sa", "mep.gov.sa",
  "moe.gov.sa", "moenergy.gov.sa", "mcit.gov.sa", "mec.gov.sa"
]

if __name__ == "__main__":
  scenarios_to_test = [
    {
      "scenario": "USA ogłaszają całkowity zakaz importu kluczowych komponentów półprzewodnikowych z Chin, co prowadzi do wstrzymania produkcji w kilku dużych niemieckich fabrykach samochodowych. Japonia zacieśnia współpracę w celu ustabilizowania łańcuchów dostaw.",
      "weight": 9
    },
    {
      "scenario": "Rosja zapowiada częściowe wstrzymanie dostaw ropy do Europy Północnej w związku z sankcjami. Finlandia i Niemcy zwołują pilne spotkanie w sprawie bezpieczeństwa energetycznego, a Arabia Saudyjska ogłasza duże inwestycje w zielony wodór.",
      "weight": 8
    },
    {
      "scenario": "Na granicy ukraińsko-rosyjskiej dochodzi do incydentu wojskowego, w wyniku którego Stany Zjednoczone ogłaszają zwiększenie obecności wojskowej w regionie. Indie i Chiny wydają wspólne, łagodzące oświadczenie.",
      "weight": 10
    }
  ]

  # Inicjalizacja agenta
  agent = ScenarioAgent()

  # Wywołanie nowej metody wsadowej
  final_batch_output = agent.process_batch(scenarios_to_test, MSZ_OFFICIAL_SOURCES)

  print("\n\n#####################################################")
  print("### ZAGREGOWANY WYNIK ANALIZY WIELU SCENARIUSZY ###")
  print("#####################################################")
  print(final_batch_output.model_dump_json(indent=2))