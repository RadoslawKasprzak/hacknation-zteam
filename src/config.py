import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
PLLUM_API_KEY: str = os.getenv("PLLUM_API_KEY")
PLLUM_BASE_URL: str = os.getenv("PLLUM_BASE_URL")
PLLUM_MODEL_NAME: str = os.getenv("PLLUM_MODEL_NAME")


CONTEXT_ATLANTIS: str = """
Nazwa państwa: Atlantis
Istotne cechy położenia geograficznego: dostęp do Morza Bałtyckiego, kilka dużych żeglownych rzek, ograniczone zasoby wody pitnej
Liczba ludności: 28 mln
Klimat: umiarkowany
Silne strony gospodarki: przemysł ciężki, motoryzacyjny, spożywczy, chemiczny, ICT, ambicje odgrywania istotnej roli w zakresie OZE, przetwarzania surowców krytycznych oraz budowy ponadnarodowej infrastruktury AI (m.in. big data centers, giga fabryki AI, komputery kwantowe)
Liczebność armii: 150 tys. zawodowych żołnierzy
Stopnień cyfryzacji społeczeństwa: powyżej średniej europejskiej
Waluta: inna niż euro
Kluczowe relacje dwustronne: Niemcy, Francja, Finlandia, Ukraina, USA, Japonia
Potencjalne zagrożenia polityczne i gospodarcze: niestabilność w UE, rozpad UE na grupy „różnych prędkości”; negatywna kampania wizerunkowa; zakłócenia w dostawach paliw węglowodorowych; narażenie na spowolnienie rozwoju sektora ICT z powodu embarga na zaawansowane procesory
Potencjalne zagrożenie militarne: zagrożenie atakiem zbrojnym jednego z sąsiadów; trwające od wielu lat ataki hybrydowe.
"""

# --- 4. SCHEMATY WYJŚCIOWE DLA LANGCHAIN ---

# Schemat dla Etapu 0: Anonimizacja i ekstrakcja URL
ANONYMIZER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "anonymized_query": {"type": "string", "description": "Anonimizowane zapytanie, w którym usunięto wrażliwe dane."},
        "urls_to_extract": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista wszystkich adresów URL/domen, które należy przeanalizować."
        }
    },
    "required": ["anonymized_query", "urls_to_extract"]
}

# Schemat dla Etapu 2: Ostateczna analiza
ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Tytuł analizy."},
        "summary": {"type": "string", "description": "Krótkie podsumowanie dla Ministra."},
        "relevance_for_atlantis": {"type": "number", "description": "Ocena istotności 0-10."},
        "full_analysis": {"type": "string", "description": "Pełna merytoryczna analiza z odniesieniem do cech Atlantis."}
    },
    "required": ["title", "summary", "relevance_for_atlantis", "full_analysis"]
}