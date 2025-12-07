import logging
import os
import json
import uuid
import threading
import time
import re
from typing import Dict, List
import datetime

# --- BIBLIOTEKI INFRASTRUKTURALNE ---
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy
import pg8000
import pdfplumber
import numpy as np
from pypdf import PdfReader

# --- BIBLIOTEKI ML/CHUNKOWANIA ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from safety_agent import safety_agent
from external_research_agent_2 import ExternalResearchAgent
from predictive_impact_agent import PredictiveImpactAgent
from summary_brief_agent import SummaryBriefAgent
from summary_report_agent import SummaryReportAgent
from scenario_agent_with_verificator import scenario_agent_with_verificator

# Ładowanie zmiennych środowiskowych z pliku .env
load_dotenv()
logging.basicConfig(level=logging.INFO)

# =======================================================
# === KONFIGURACJA ŚRODOWISKA I KLIENTÓW ===
# =======================================================

PORT = int(os.environ.get("PORT", 8080))
HOME_COUNTRY_NAME = "Atlantis"

# Zmienne środowiskowe GCP/OpenAI
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
SQL_CONNECTION_NAME = os.environ.get("SQL_CONNECTION_NAME")
SQL_DATABASE = os.environ.get("SQL_DATABASE")
SQL_USER = os.environ.get("SQL_USER")
SQL_PASSWORD = os.environ.get("SQL_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Konieczne dla klienta OpenAI

DB_ENGINE = None
research_queue = {}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    client_openai = OpenAI()
    client_gcs = storage.Client()
    connector = Connector()
except Exception as e:
    logging.error(f"Błąd inicjalizacji klientów usług: {e}")
    # Aplikacja wystartuje, ale operacje na chmurze będą niemożliwe

app = Flask(__name__)


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
                            "motoryzacyjny będzie miał w roku 2025 zyski na poziomie 30% średnich rocznych zysków z lat 2020-2024",
                            15)])


# =======================================================
# === FUNKCJE DB I EMBEDDINGU (LangChain, GCS, SQL) ===
# =======================================================

def init_db_engine():
    global DB_ENGINE
    if DB_ENGINE is not None: return DB_ENGINE
    try:
        def getconn() -> pg8000.dbapi.Connection:
            conn = connector.connect(SQL_CONNECTION_NAME, "pg8000", user=SQL_USER, password=SQL_PASSWORD, db=SQL_DATABASE, ip_type=IPTypes.PUBLIC)
            return conn
        DB_ENGINE = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn, pool_pre_ping=True, pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)
        return DB_ENGINE
    except Exception as e:
        logging.error(f"Błąd inicjalizacji Engine'u SQLAlchemy: {e}"); DB_ENGINE = None; raise e

def get_chunks_from_text(text, chunk_size=1000, overlap=100):
    if not text: return []
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len, separators=["\n\n", "\n", " ", ""])
    return text_splitter.split_text(cleaned_text)

def generate_embedding(text_content):
    response = client_openai.embeddings.create(input=text_content, model="text-embedding-ada-002")
    return response.data[0].embedding

def save_embedding_to_db(engine, filename, chunk_text, embedding_vector):
    vector_string = '[' + ','.join(map(str, embedding_vector)) + ']'
    insert_query_template = f"""
        INSERT INTO documents (filename, content, embedding, created_at) 
        VALUES (:filename, :content, '{vector_string}'::vector, now())
    """
    insert_query = sqlalchemy.text(insert_query_template)
    with engine.connect() as conn:
        conn.execute(insert_query, {"filename": filename, "content": chunk_text}); conn.commit()

def check_if_file_exists_in_db(filename):
    engine = init_db_engine()
    select_query = sqlalchemy.text("SELECT COUNT(*) FROM documents WHERE filename = :filename")
    try:
        with engine.connect() as conn:
            result = conn.execute(select_query, {"filename": filename}).scalar()
        return result > 0
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania istnienia pliku w DB: {e}"); return False


def run_engine(scenarios_raw, textfiles):
    external_agent = ExternalResearchAgent()
    predictive_agent = PredictiveImpactAgent()
    summary_agent = SummaryReportAgent()
    brief_agent = SummaryBriefAgent()

    all_external_results_per_scenario = []

    for scenario, weight in scenarios_raw:
        logging.info(f"Rozpoczynanie analizy scenariusza (waga={weight}): {scenario[:50]}...")

        resp = scenario_agent_with_verificator(user_prompt, scenario, weight)

        if isinstance(resp, dict) and "countries" in resp and "subjects" in resp:
            countries = resp["countries"]
            subjects = resp["subjects"]
        else:
            logging.error(f"❌ BŁĘDNA STRUKTURA z scenario_agent dla: {scenario[:30]}")
            continue

        sanitized_user_prompt, sanitized_scenario = safety_agent(user_prompt, scenario)

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

        all_external_results_per_scenario.append({
            "scenario": scenario,
            "weight": weight,
            "countries": countries,
            "subjects": subjects,
            "external_results": external_results,
            "predictions": predictions,
        })

    # ZAPIS SUROWYCH DANYCH
    with open("external_results.json", "w", encoding="utf-8") as f:
        json.dump(all_external_results_per_scenario, f, ensure_ascii=False, indent=2)

    # RAPORT ZBIORCZY
    final_report = summary_agent.build_global_report(
        home_context=user_prompt,
        scenarios_data=all_external_results_per_scenario,
    )


    # KRÓTKIE STRESZCZENIE (250–300 słów)
    brief_summary = brief_agent.build_brief_summary(final_report)
    with open("raport_atlantis.md", "w", encoding="utf-8") as f:
        f.write(final_report)

    return {
        "status": "completed",
        "raw_data": all_external_results_per_scenario,
        "final_report": final_report,
        "brief_summary": brief_summary,
        "context_files_used": textfiles,
    }


def embed_chunks_to_db_worker(file_id, original_filename, text_content):
    job = research_queue.get(file_id)
    if job:
        job['status'] = 'processing_embedding'

    logging.info(f"Start przetwarzania embeddingu dla pliku: {original_filename}")

    try:
        if GCS_BUCKET_NAME:
            logging.warning("Pomięto zapis do GCS - brakuje implementacji GCS w tym bloku kodu.")

        chunks = get_chunks_from_text(text_content, chunk_size=1000, overlap=100)
        logging.info(f"Plik podzielony na {len(chunks)} fragmentów (LangChain).")

        engine = init_db_engine()

        for i, chunk in enumerate(chunks):
            embedding_vector = generate_embedding(chunk)
            save_embedding_to_db(engine, original_filename, chunk, embedding_vector)
            job['progress'] = f"{i + 1}/{len(chunks)}"

        set_to_done(file_id, {"chunks_processed": len(chunks), "filename": original_filename})
        logging.info(f"Zakończono zapis embeddingów dla pliku: {original_filename}")

    except Exception as e:
        logging.error(f"Krytyczny błąd worker'a embeddingu dla {original_filename}: {e}")
        research_queue[file_id]['status'] = 'error'
        research_queue[file_id]['error'] = str(e)


def pass_research_request(research_id, scenarios, textfiles):
    def worker():
        research_queue[research_id]['status'] = 'running'
        try:
            res = run_engine(scenarios, textfiles)
            set_to_done(research_id, res)
        except Exception as e:
            research_queue[research_id]['status'] = 'error'
            research_queue[research_id]['error'] = str(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def set_to_done(research_id, result=None):
    job = research_queue.get(research_id)
    if job is not None:
        job['status'] = 'done'
        if result is not None:
            job['result'] = result



@app.route('/', methods=['GET'])
def index():
    try:
        # Pamiętaj, aby plik index.html znajdował się w podkatalogu 'templates'
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Błąd renderowania szablonu: {e}")
        return "Witaj! (Błąd ładowania UI. Sprawdź, czy 'templates/index.html' istnieje)", 500


@app.route('/status', methods=['GET'])
def get_status():
    research_id = request.args.get('research_id')
    if not research_id:
        return jsonify({'error': 'Missing research_id query parameter'}), 400
    job = research_queue.get(research_id)
    if not job:
        return jsonify({'error': 'Research id not found'}), 404

    result = job.get('result', {})

    # Zwracamy wynik, który zawiera skrót raportu, gdy status jest 'done'
    brief_summary_display = result.get('brief_summary', 'Analiza w toku...') if job.get('status') == 'done' else None

    return jsonify({
        'research_id': research_id,
        'status': job.get('status'),
        'progress': job.get('progress', 'N/A'),
        'result': brief_summary_display
    }), 200


@app.route('/research', methods=['POST'])
def start_research():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Missing or invalid JSON body'}), 400

    scenarios_data = data.get('scenarios')
    context_files = data.get('context_files')

    if not scenarios_data:
        return jsonify({'error': 'Missing scenarios in the request body'}), 400

    if context_files is None:
        context_files = []

    # Pobieranie treści plików kontekstowych (które zostały załadowane do UPLOAD_DIR)
    textfiles = []
    for file_id in context_files:
        txt_path = os.path.join(UPLOAD_DIR, file_id + ".txt")
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                textfiles.append(f.read())
        except FileNotFoundError:
            logging.warning(f"Plik kontekstowy {file_id}.txt nie został znaleziony.")
            continue

    research_id = uuid.uuid4().hex
    job = {
        'id': research_id,
        'status': 'queued',
        'result': None,
    }
    research_queue[research_id] = job

    # Uruchomienie workera
    pass_research_request(research_id, scenarios_data, textfiles)

    return jsonify({'research_id': research_id, 'status': 'queued'}), 202


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request (expected key "file")'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    original_filename = file.filename
    file_extension = original_filename.split(".")[-1].lower()

    if file_extension not in ['pdf', 'txt']:
        return jsonify({'error': 'Nieobsługiwany format pliku (tylko PDF/TXT)'}), 400

    # WALIDACJA DUPLIKATÓW
    if check_if_file_exists_in_db(original_filename):
        return jsonify({
            'error': f'Plik o nazwie "{original_filename}" został już przetworzony i jest w bazie danych.',
            'status': 'conflict'
        }), 409
    if any(job.get('original_filename') == original_filename and job.get('status') in ['queued', 'processing_embedding']
           for job in research_queue.values()):
        return jsonify({
            'error': f'Plik o nazwie "{original_filename}" jest już w trakcie przetwarzania.',
            'status': 'processing'
        }), 409

    # 1. Generowanie ID i zapis pliku tymczasowego
    file_id = uuid.uuid4().hex
    temp_save_path = os.path.join(UPLOAD_DIR, file_id + "." + file_extension)
    file.save(temp_save_path)

    text_content = ""

    # 2. Wyodrębnianie tekstu
    if file_extension == "pdf":
        try:
            with pdfplumber.open(temp_save_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_content += t + '\n'
        except Exception as e:
            os.remove(temp_save_path)
            return jsonify({'error': f'Błąd przetwarzania PDF: {str(e)}'}), 500
    else:
        with open(temp_save_path, "r", encoding="utf-8") as f:
            text_content = f.read()

    # ZAPISUJEMY TEKST W PLIKU .TXT dla /research endpointu, a następnie usuwamy plik tymczasowy.
    with open(os.path.join(UPLOAD_DIR, file_id + ".txt"), "w", encoding="utf-8") as f:
        f.write(text_content)
    os.remove(temp_save_path)

    # 3. Uruchomienie asynchronicznego zadania chunkowania i embeddingu
    job = {
        'id': file_id,
        'status': 'queued',
        'original_filename': original_filename,
        'progress': '0/0'
    }
    research_queue[file_id] = job

    t = threading.Thread(target=embed_chunks_to_db_worker, args=(file_id, original_filename, text_content), daemon=True)
    t.start()

    return jsonify({
        'file_id': file_id,
        'original_filename': original_filename,
        'status': 'embedding_queued',
        'check_status_url': f'/status?research_id={file_id}'
    }), 202


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)