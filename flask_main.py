import logging
import os
import datetime
import io
import uuid
import threading
import time
import re  # Nowy import do czyszczenia tekstu
from pypdf import PdfReader
import pdfplumber
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from google.cloud import storage
from google.cloud.sql.connector import Connector, IPTypes
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sqlalchemy
import pg8000


load_dotenv()
logging.basicConfig(level=logging.INFO)

PORT = int(os.environ.get("PORT", 8080))

# --- KONFIGURACJA ZMIENNYCH ŚRODOWISKOWYCH ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
SQL_CONNECTION_NAME = os.environ.get("SQL_CONNECTION_NAME")
SQL_DATABASE = os.environ.get("SQL_DATABASE")
SQL_USER = os.environ.get("SQL_USER")
SQL_PASSWORD = os.environ.get("SQL_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- INICJALIZACJA KLIENTÓW I KATALOGÓW ---
DB_ENGINE = None
research_queue = {}  # Kolejka do śledzenia asynchronicznych zadań
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Tworzenie katalogu uploads

try:
    client_openai = OpenAI()
    client_gcs = storage.Client()
    connector = Connector()
except Exception as e:
    logging.error(f"Błąd inicjalizacji klientów usług: {e}")

app = Flask(__name__)


# --- FUNKCJE BAZY DANYCH I EMBEDDINGU (Poprawione i uzupełnione) ---

def init_db_engine():
    """Tworzy instancję engine'u SQLAlchemy."""
    global DB_ENGINE
    if DB_ENGINE is not None:
        return DB_ENGINE

    # ... (kod init_db_engine bez zmian, jest poprawny) ...
    try:
        def getconn() -> pg8000.dbapi.Connection:
            conn = connector.connect(
                SQL_CONNECTION_NAME,
                "pg8000",
                user=SQL_USER,
                password=SQL_PASSWORD,
                db=SQL_DATABASE,
                ip_type=IPTypes.PUBLIC
            )
            return conn

        DB_ENGINE = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=1800
        )
        return DB_ENGINE

    except Exception as e:
        logging.error(f"Błąd inicjalizacji Engine'u SQLAlchemy: {e}")
        DB_ENGINE = None
        raise e


def get_chunks_from_text(text, chunk_size=1000, overlap=100):
    """
    Dzieli tekst na fragmenty za pomocą LangChain RecursiveCharacterTextSplitter.
    Używa chunk_size (domyślnie 1000 znaków) i overlap (domyślnie 100 znaków).
    """
    if not text:
        return []

    # Oczyszczenie tekstu
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    # Inicjalizacja LangChain Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Dzielenie tekstu
    chunks = text_splitter.split_text(cleaned_text)

    return chunks


# ZMIENIONA NAZWĘ FUNKCJI, ABY BYŁA ZGODNA Z NAWYMI WYWOŁANIAMI
def generate_embedding(text_content):
    """Generuje wektor embeddingu za pomocą OpenAI."""
    response = client_openai.embeddings.create(
        input=text_content,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


def save_embedding_to_db(engine, filename, chunk_text, embedding_vector):
    """Zapisuje embedding do tabeli 'documents'."""
    vector_string = '[' + ','.join(map(str, embedding_vector)) + ']'

    insert_query_template = f"""
        INSERT INTO documents (filename, content, embedding, created_at) 
        VALUES (:filename, :content, '{vector_string}'::vector, now())
    """
    insert_query = sqlalchemy.text(insert_query_template)

    with engine.connect() as conn:
        conn.execute(insert_query, {
            "filename": filename,
            "content": chunk_text,
        })
        conn.commit()


# --- FUNKCJE ASYNCHRONICZNE (Long Running Task) ---

def pass_research_request_to_engine(user_prompt, scenarios, textfiles):
    time.sleep(1)
    return {"message": "Simulacja researchu zakończona."}


def embed_chunks_to_db_worker(file_id, original_filename, text_content):
    """Worker do przetwarzania pliku, chunkowania, zapisu do GCS i bazy."""
    job = research_queue.get(file_id)
    if job:
        job['status'] = 'processing_embedding'

    logging.info(f"Start przetwarzania embeddingu dla pliku: {original_filename}")

    gcs_blob_name = None  # Inicjalizacja ścieżki GCS na wypadek błędu

    try:
        # --- NOWA LOGIKA: ZAPIS TREŚCI TEKSTOWEJ DO GCS ---
        if not GCS_BUCKET_NAME:
            raise Exception("Brak zmiennej GCS_BUCKET_NAME, nie można zapisać w Storage.")

        bucket = client_gcs.bucket(GCS_BUCKET_NAME)
        # Tworzymy unikalną ścieżkę do archiwum tekstu
        gcs_blob_name = f"archived_text/{file_id}_{original_filename.replace('.', '_')}.txt"
        blob = bucket.blob(gcs_blob_name)

        # Zapisujemy wyodrębnioną treść tekstową (która jest w pamięci) bezpośrednio do GCS
        blob.upload_from_string(text_content, content_type='text/plain')
        logging.info(f"Pomyślnie zapisano plik tekstowy w GCS jako: {gcs_blob_name}")
        # ----------------------------------------------------------------------

        # 1. Chunkowanie tekstu
        chunks = get_chunks_from_text(text_content, chunk_size=1000, overlap=100)
        logging.info(f"Plik podzielony na {len(chunks)} fragmentów (LangChain).")

        engine = init_db_engine()

        # 2. Iteracja, generowanie embeddingów i zapis do DB
        for i, chunk in enumerate(chunks):
            logging.info(f"Generowanie embeddingu dla fragmentu {i + 1}/{len(chunks)}...")
            embedding_vector = generate_embedding(chunk)

            # Zapis do bazy
            save_embedding_to_db(engine, original_filename, chunk, embedding_vector)
            job['progress'] = f"{i + 1}/{len(chunks)}"

        # 3. Zakończenie pracy
        set_to_done(file_id, {
            "chunks_processed": len(chunks),
            "filename": original_filename,
            "gcs_path": gcs_blob_name  # Dodajemy ścieżkę do wyniku
        })
        logging.info(f"Zakończono zapis embeddingów i GCS dla pliku: {original_filename}")

    except Exception as e:
        logging.error(f"Krytyczny błąd worker'a embeddingu dla {original_filename}: {e}")
        research_queue[file_id]['status'] = 'error'
        research_queue[file_id]['error'] = str(e)


def set_to_done(research_id, result=None):
    job = research_queue.get(research_id)
    if job is not None:
        job['status'] = 'done'
        if result is not None:
            job['result'] = result


def check_if_file_exists_in_db(filename):
    """Sprawdza, czy plik o danej nazwie już istnieje w tabeli documents."""
    engine = init_db_engine()

    select_query = sqlalchemy.text("SELECT COUNT(*) FROM documents WHERE filename = :filename")

    try:
        with engine.connect() as conn:
            # Używamy .scalar() aby uzyskać samą wartość z wyniku
            result = conn.execute(select_query, {"filename": filename}).scalar()

        return result > 0
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania istnienia pliku w DB: {e}")
        # Jeśli wystąpi błąd DB, traktujemy to jako brak duplikatu, aby nie blokować uploadu,
        # ale logujemy błąd. W produkcji można rzucić błąd 500.
        return False
# --- ENDPOINTY ZINTEGROWANE ---

@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception:
        return "Witaj! (Aplikacja zintegrowana)."


@app.route('/storage', methods=['GET'])
def list_bucket_contents():
    # ... (kod list_bucket_contents bez zmian) ...
    if not GCS_BUCKET_NAME:
        return jsonify({"error": "Brak zmiennej GCS_BUCKET_NAME"}), 500

    try:
        bucket = client_gcs.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(max_results=10)

        file_list = [{"name": blob.name, "size": blob.size, "updated": blob.updated.isoformat()}
                     for blob in blobs]

        return jsonify({"bucket": GCS_BUCKET_NAME, "files": file_list}), 200

    except Exception as e:
        logging.error(f"Błąd GCS: {e}")
        return jsonify({"error": f"Błąd GCS: {str(e)}"}), 500


@app.route('/sql', methods=['GET'])
def get_sql_data():
    # ... (kod get_sql_data bez zmian) ...
    try:
        if not all([SQL_CONNECTION_NAME, SQL_USER, SQL_PASSWORD, SQL_DATABASE]):
            return jsonify({"error": "Brakuje zmiennych środowiskowych do połączenia z SQL"}), 500

        engine = init_db_engine()

        with engine.connect() as conn:
            result = conn.execute(sqlalchemy.text("SELECT now();")).fetchone()
            database_time = result[0]
            if isinstance(database_time, datetime.datetime):
                time_str = database_time.isoformat()
            else:
                time_str = str(database_time)

            return jsonify({
                "database_time": time_str,
                "operation": "SELECT now() (via SQLAlchemy)",
                "status": "success"
            }), 200

    except Exception as e:
        logging.error(f"Błąd Cloud SQL: {e}")
        return jsonify({"error": f"Błąd Cloud SQL: {str(e)}"}), 500


# Endpoint do asynchronicznego sprawdzania statusu
@app.route('/status', methods=['GET'])
def get_status():
    research_id = request.args.get('research_id')
    if not research_id:
        return jsonify({'error': 'Missing research_id query parameter'}), 400
    job = research_queue.get(research_id)
    if not job:
        return jsonify({'error': 'Research id not found'}), 404
    return jsonify(
        {'research_id': research_id, 'status': job.get('status'), 'progress': job.get('progress', 'N/A')}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(
            {'error': 'No file part in the request (expected key "file")'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    original_filename = file.filename
    file_extension = original_filename.split(".")[-1].lower()

    if file_extension not in ['pdf', 'txt']:
        return jsonify({'error': 'Nieobsługiwany format pliku (tylko PDF/TXT)'}), 400

    # -----------------------------------------------------------------
    # NOWA WALIDACJA DUPLIKATÓW
    # -----------------------------------------------------------------

    # 1. Sprawdzenie, czy plik jest już w bazie
    if check_if_file_exists_in_db(original_filename):
        return jsonify({
            'error': f'Plik o nazwie "{original_filename}" został już przetworzony i jest w bazie danych.',
            'status': 'conflict'
        }), 409

    # 2. Sprawdzenie, czy plik jest już w trakcie przetwarzania (w kolejce in-memory)
    if any(job.get('original_filename') == original_filename and job.get('status') in ['queued', 'processing_embedding']
           for job in research_queue.values()):
        return jsonify({
            'error': f'Plik o nazwie "{original_filename}" jest już w trakcie przetwarzania.',
            'status': 'processing'
        }), 409

    # -----------------------------------------------------------------

    # 1. Generowanie ID i zapis pliku tymczasowego
    file_id = uuid.uuid4().hex
    temp_save_path = os.path.join(UPLOAD_DIR, file_id + "." + file_extension)
    file.save(temp_save_path)

    # ... (reszta logiki przetwarzania pliku pozostaje bez zmian) ...

    text_content = ""

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


#
# Poniższe endpointy są zachowane tylko jako symulacja pierwotnej logiki "research"
#
@app.route('/research', methods=['POST'])
def start_research():
    # Endpoint symulujący start dłuższego zadania
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Missing or invalid JSON body'}), 400

    # ... (usuń logikę ładowania plików z UPLOAD_DIR, ponieważ teraz upload_file to robi) ...

    # Create research job and enqueue
    research_id = uuid.uuid4().hex
    job = {
        'id': research_id,
        'status': 'queued'
    }
    research_queue[research_id] = job

    # Uruchomienie worker'a w osobnym wątku (do symulacji)
    def worker():
        research_queue[research_id]['status'] = 'running'
        try:
            res = pass_research_request_to_engine(data.get('user_prompt'), data.get('scenarios'), [])
            set_to_done(research_id, res)
        except Exception as e:
            research_queue[research_id]['status'] = 'error'
            research_queue[research_id]['error'] = str(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    return jsonify({'research_id': research_id, 'status': 'queued'}), 202


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)