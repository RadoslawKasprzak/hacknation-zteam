from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

import config2


class TopicAnalysis(BaseModel):
  """Struktura pojedynczego dopasowania tematyki do domen."""
  topic: str = Field(..., description="Jedna z dozwolonych tematyk.")
  selected_domains: List[str] = Field(...,
    description="Lista domen z whitelisty, które są najbardziej relewantne dla tej tematyki.")
  # NOWE POLE 1: Podtematy
  subtopics: List[str] = Field(...,
                               description="Lista dokładnie 3 szczegółowych podtematów wynikających ze scenariusza dla tej kategorii.")

  # NOWE POLE 2: Prompt do wyszukiwarki
  search_query_prompt: str = Field(...,
                                   description="Sformułowane zapytanie/prompt, który pozwoli na skuteczne wyszukanie informacji o tych podtematach w kontekście Polski.")

class ScenarioOutput(BaseModel):
  """Główny obiekt wyjściowy agenta."""
  identified_topics: List[str] = Field(..., description="Lista wykrytych tematów (max 5).")
  domain_mapping: List[TopicAnalysis] = Field(..., description="Mapowanie tematów na wybrane domeny.")
  metadata_cot: Dict[str, str] = Field(...,
                                       description="Zbiór metadanych i wewnętrznych uzasadnień modelu dot. analizy, w "
                                                   "tym: Wpływ Scenariusza na Polskę, Uzasadnienie Wyboru Domen.")

class BatchScenarioAnalysis(BaseModel):
    """Główny obiekt przechowujący analizę wielu scenariuszy."""
    analysis_batch: List[ScenarioOutput] = Field(..., description="Lista obiektów ScenarioOutput, gdzie każdy reprezentuje analizę jednego scenariusza.")

class ScenarioAgent:
  def __init__(self, model_name="gpt-4o"):
    # Inicjalizacja modelu (zalecany GPT-4o dla lepszego rozumowania w jęz. polskim)
    self.llm = ChatOpenAI(model="gpt-4.1", api_key=lambda: config2.OPENAI_API_KEY)
#       ChatOpenAI(
#             model="XXXXXXXXXXXXXXXXXXXXXXX",
#             openai_api_key="EMPTY",
#             openai_api_base="XXXXXXXXXXXXXXXXXXXXXXX",
#             default_headers={
#         'Ocp-Apim-Subscription-Key': "XXXXXXXXXXXXXXXXXXX"
#     }
# )

    self.parser = PydanticOutputParser(pydantic_object=ScenarioOutput)

  def _create_prompt(self):
    """Tworzy szablon promptu z instrukcjami dla analityka MSZ."""
    system_template = """
        Jesteś ekspertem analitycznym w Ministerstwie Spraw Zagranicznych RP.
        Twoim zadaniem jest wstępna analiza scenariusza geopolitycznego pod kątem jego wpływu na Polskę.

        Kontekst analizy obejmuje relacje z kluczowymi partnerami:
        Japonia, Ukraina, USA, Niemcy, Finlandia, Francja.

        Twoje zadanie składa się z dwóch kroków:
        1. Kategoryzacja: Przypisz scenariusz do maksymalnie 5 kategorii z poniższej listy:
           - Gospodarka i Handel Międzynarodowy
           - Energetyka i Surowce Strategiczne
           - Technologie i Przemysł
           - Geopolityka i Bezpieczeństwo
           - Klimat i Środowisko

        2. Dobór Źródeł: Dla każdej wybranej kategorii wybierz z podanej 'Whitelisty Domen' te serwisy,
           które będą najlepsze do pogłębionego researchu w internecie dla tego konkretnego tematu.
           UWAGA: Możesz wybierać TYLKO domeny znajdujące się na przekazanej whieliście.
        
        3. Podtematy: Do każdej kategorii dobierz 3 konkretne podtematy.
          Przykład: 
          Dane Giełdowe i Rynki Kapitałowe:
            Notowania lokalnych spółek technologicznych;
            Indeksy surowcowe (Metale ziem rzadkich, Krzem)
            Notowania konkurencji.
            
        4. Search Prompt: Napisz do każdej kategorii JEDNO precyzyjne zapytanie (prompt) do wyszukiwarki.
        
        5. Metadane/CoT (Wewnętrzne Uzasadnienie): Przygotuj kluczowe uzasadnienia dla swojej decyzji.
        Waga scenariusza: {weight} (gdzie wyższa waga oznacza wyższy priorytet i konieczność dokładniejszej selekcji źródeł).
        """

    human_template = """
        SCENARIUSZ: {scenario}

        WHITELISTA DOMEN:
        {domain_whitelist}

        {format_instructions}
        """

    return ChatPromptTemplate.from_messages([
      ("system", system_template),
      ("human", human_template)
    ])

  def analyze(self, scenario: str, weight: int, domain_whitelist: List[str]) -> ScenarioOutput:
    """
    Główna metoda wywołująca agenta.
    """
    prompt = self._create_prompt()

    chain = prompt | self.llm.with_structured_output(ScenarioOutput)

    base_result = chain.invoke({
      "scenario": scenario,
      "weight": weight,
      "domain_whitelist": ", ".join(domain_whitelist),
      "format_instructions": self.parser.get_format_instructions()
    })

    base_result.original_scenario_input = scenario
    base_result.scenario_weight = weight

    return base_result

  def process_batch(self, scenarios_data: List[Dict], domain_whitelist: List[str]) -> BatchScenarioAnalysis:
      """
      Wykonuje analizę wsadową (batch) dla listy scenariuszy.

      Args:
          scenarios_data: Lista słowników, każdy z kluczami 'scenario' (str) i 'weight' (int).

      Returns:
          Obiekt BatchScenarioAnalysis zawierający listę wyników ScenarioOutput.
      """

      all_analysis_results = []
      print(f"--- Rozpoczęcie przetwarzania wsadowego {len(scenarios_data)} scenariuszy ---")

      for i, data in enumerate(scenarios_data):
          scenario = data.get('scenario', 'Brak scenariusza')
          weight = data.get('weight', 5)

          print(f"[{i + 1}/{len(scenarios_data)}] Analiza: {scenario[:70]}... (Waga: {weight})")

          try:
              # Wywołanie pojedynczej analizy
              result = self.analyze(scenario=scenario,
                                    weight=weight,
                                    domain_whitelist=domain_whitelist)
              all_analysis_results.append(result)

          except Exception as e:
              print(f"⚠️ Błąd podczas analizy scenariusza {i + 1}: {e}")
              # Można dodać logowanie błędu lub dodanie "pustego" wyniku dla zachowania ciągłości batcha

      print("\n--- Przetwarzanie wsadowe zakończone ---")

      # Zwracamy główny obiekt kontenerowy
      return BatchScenarioAnalysis(analysis_batch=all_analysis_results)
