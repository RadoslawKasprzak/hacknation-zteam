import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

actual_scenario = """Jesteś analitykiem bezpieczeństwa dla fikcyjnego kraju Atlantis. W JSON w którym dostaniesz, usuń TYLKO poufne. Wysyłaj spowrotem TYLKO JSON, na podstawie [JSON TEMPLATE]\n\n.

[DANE PAŃSTWA]
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

[JSON TEMPLATE]
{
  "analysis_timestamp": "",
  "scenario_id": "",
  "importance_weight": "",

  "analysis_notes": "",

  "global_areas": [
    {
      "area_name": "",
      "scenario_impact_summary": "",

      "country_impact_analysis": [
        { "country": "" }
      ],

      "information_sources": [
        {
          "source_name": "",
          "url_or_description": "",
          "source_type": ""
        }
      ]
    },
    {
      "area_name": "",
      "scenario_impact_summary": "",

      "country_impact_analysis": [
        { "country": "" }
      ],

      "information_sources": [
        {
          "source_name": "",
          "url_or_description": "",
          "source_type": ""
        }
      ]
    },
    {
      "area_name": "",
      "scenario_impact_summary": "",

      "country_impact_analysis": [
        { "country": "" }
      ],

      "information_sources": [
        {
          "source_name": "",
          "url_or_description": "",
          "source_type": ""
        }
      ]
    },
    {
      "area_name": "",
      "scenario_impact_summary": "",

      "country_impact_analysis": [
        { "country": "" }
      ],

      "information_sources": [
        {
          "source_name": "",
          "url_or_description": "",
          "source_type": ""
        }
      ]
    },
    {
      "area_name": "",
      "scenario_impact_summary": "",

      "country_impact_analysis": [
        { "country": "" }
      ],

      "environmental_notes": "",

      "information_sources": [
        {
          "source_name": "",
          "url_or_description": "",
          "source_type": ""
        }
      ]
    }
  ]
}

"""

api_key = "API_KEY"
model_name = "gpt-4.1"


llm = ChatOpenAI(
    model=model_name,
    api_key=lambda: api_key,
    # openai_api_base=base_url,
    temperature=0.7,
    max_tokens=20000
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json("mock_prompt.json")

def safety_agent(super_prompt):
    response = llm.invoke(actual_scenario, super_prompt)
    return json.loads(response.model_dump_json())['content']