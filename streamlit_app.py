import streamlit as st
import requests
import json
import time
import os

# --- KONFIGURACJA ---
# Adres API Twojej aplikacji Flask (musi działać w tle!)
FLASK_API_URL = "http://127.0.0.1:8080"
RESEARCH_QUEUE_TIMEOUT = 120  # Sekundy oczekiwania na zakończenie zadania

# Ustawienie nagłówków strony Streamlit
st.set_page_config(page_title="Atlantis Research Agent", layout="wide")


# =======================================================
# === FUNKCJE API ===
# =======================================================

@st.cache_data
def load_default_scenarios():
    """Wczytuje domyślny zestaw scenariuszy do pola tekstowego."""
    return """
[
    ["Wskutek zaistniałej przed miesiącem katastrofy naturalnej wiodący światowy producent procesorów graficznych stracił 60% zdolności produkcyjnych; odbudowa mocy produkcyjnych poprzez inwestycje w filie zlokalizowane na obszarach nieobjętych katastrofą potrwa do końca roku 2028", 30],
    ["Przemysł motoryzacyjny w Europie (...) bardzo wolno przestawia się na produkcję samochodów elektrycznych; rynek europejski zalewają tanie samochody elektryczne z Azji Wschodniej...", 15]
]
"""


def api_post(endpoint, data=None, files=None):
    """Ogólna funkcja do wywoływania API Flaska."""
    url = f"{FLASK_API_URL}{endpoint}"
    try:
        if files:
            response = requests.post(url, files=files)
        else:
            response = requests.post(url, json=data)

        response.raise_for_status()  # Rzuca wyjątek dla kodów 4xx/5xx
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Błąd API {endpoint}: {e.response.status_code}")
        try:
            return e.response.json()
        except:
            return {"error": f"Nieznany błąd serwera. Status: {e.response.status_code}"}
    except Exception as e:
        st.error(f"Błąd połączenia: Upewnij się, że serwer Flask działa na {FLASK_API_URL}. Szczegóły: {e}")
        return {"error": str(e)}


# =======================================================
# === INTERFEJS UŻYTKOWNIKA STREAMLIT ===
# =======================================================

def main():
    st.title("🛡️ System Analizy Ryzyka Atlantis (AI Agent)")
    st.markdown("---")

    # --- 1. UPLOAD I EMBEDDING ---
    st.header("1. Kontekst: Ładowanie pliku i Embedding")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Wybierz plik PDF lub TXT do wgrania:", type=['pdf', 'txt'])

    if uploaded_file:
        if st.button("Uruchom Embedding (POST /upload)", key="upload_btn"):
            with st.spinner(f"Przetwarzanie i chunkowanie '{uploaded_file.name}'..."):
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                result = api_post("/upload", files=files)

                if result.get('file_id'):
                    st.session_state['file_id'] = result['file_id']
                    st.session_state['file_name'] = result['original_filename']
                    st.success(f"Plik '{result['original_filename']}' wgrany.")
                    st.info(f"FILE ID: **{result['file_id']}** (Zadanie embeddingu w tle).")

                    # Oczekiwanie na zakończenie embeddingu
                    check_embedding_status(result['file_id'], col2)
                else:
                    st.error(f"Błąd ładowania: {result.get('error', 'Nieznany błąd.')}")

    st.markdown("---")

    # --- 2. ANALIZA AGENTÓW ---
    st.header("2. Analiza Scenariuszy (Uruchomienie Agentów)")

    scenarios_input = st.text_area(
        "Scenariusze i Wagi (JSON Array):",
        value=load_default_scenarios(),
        height=200,
        key="scenarios_input"
    )

    file_id_context = st.text_input(
        "ID pliku kontekstowego (opcjonalnie):",
        value=st.session_state.get('file_id', ''),
        help="Jeśli wgrano plik w sekcji 1, to pole jest automatycznie wypełniane."
    )

    if st.button("Uruchom Analizę Agentów (POST /research)", key="research_btn"):
        try:
            scenarios_data = json.loads(scenarios_input)
            context_files = [file_id_context] if file_id_context else []

            payload = {
                "scenarios": scenarios_data,
                "context_files": context_files
            }

            result = api_post("/research", data=payload)

            if result.get('research_id'):
                st.session_state['research_id'] = result['research_id']
                st.success(f"Zadanie analityczne uruchomione. RESEARCH ID: **{result['research_id']}**")
            else:
                st.error(f"Błąd uruchamiania analizy: {result.get('error', 'Nieznany błąd.')}")

        except json.JSONDecodeError:
            st.error("Błąd: Scenariusze nie są poprawnym formatem JSON.")
        except Exception as e:
            st.exception(e)

    st.markdown("---")

    # --- 3. STATUS I RAPORT KOŃCOWY ---
    st.header("3. Monitorowanie Statusu i Raport Końcowy")

    status_id = st.text_input(
        "ID Zadania (Research ID):",
        value=st.session_state.get('research_id', ''),
        key="status_id_input"
    )

    if st.button("Sprawdź Status i Pobierz Raport (GET /status)", key="status_btn"):
        if not status_id:
            st.warning("Wprowadź ID zadania z sekcji 2.")
            return

        check_research_status(status_id)


# =======================================================
# === FUNKCJE STATUSU I MONITOROWANIA ===
# =======================================================

def get_status_api(research_id):
    """Pobiera status z endpointu /status."""
    url = f"{FLASK_API_URL}/status?research_id={research_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "error", "error_details": "Błąd połączenia lub API"}


def check_embedding_status(file_id, column):
    """Monitoruje status zadania embeddingu i aktualizuje kolumnę."""
    start_time = time.time()

    with column:
        status_placeholder = st.empty()

        while time.time() - start_time < RESEARCH_QUEUE_TIMEOUT:
            status_data = get_status_api(file_id)
            status = status_data.get('status')
            progress = status_data.get('progress', 'N/A')

            if status == 'processing_embedding':
                status_placeholder.info(f"Status Embeddingu: W TRAKCIE (Postęp: {progress})")
            elif status == 'done':
                status_placeholder.success("✅ Embedding i zapis do DB ZAKOŃCZONY.")
                return
            elif status == 'error':
                status_placeholder.error(
                    f"❌ BŁĄD Embeddingu: {status_data.get('error_details', 'Sprawdź logi serwera.')}")
                return

            time.sleep(3)  # Oczekiwanie 3 sekundy przed kolejnym zapytaniem

        status_placeholder.warning("⚠️ Przekroczono czas oczekiwania na zakończenie embeddingu.")


def check_research_status(research_id):
    """Monitoruje status zadania analitycznego i wyświetla raport końcowy."""
    start_time = time.time()
    status_placeholder = st.empty()

    while time.time() - start_time < RESEARCH_QUEUE_TIMEOUT:
        status_data = get_status_api(research_id)
        status = status_data.get('status')

        if status == 'running':
            status_placeholder.info("Status Analizy: W TRAKCIE...")
        elif status == 'done':
            brief = status_data.get('result', 'Brak brief_summary w wyniku.')
            st.subheader("✅ RAPORT KOŃCOWY (Skrót)")
            st.code(brief, language='markdown')
            status_placeholder.empty()
            return
        elif status == 'error':
            st.error(f"❌ BŁĄD Analizy: {status_data.get('error_details', 'Sprawdź logi serwera.')}")
            status_placeholder.empty()
            return

        if status == 'done':
            status_placeholder.link_button("Gotowe, pobierz raport", url=FLASK_API_URL+"/download_report")
        else:
            status_placeholder.info(f"Status: {status}")

        time.sleep(5)

    st.warning("⚠️ Przekroczono czas oczekiwania na zakończenie analizy.")


if __name__ == '__main__':
    if 'research_id' not in st.session_state:
        st.session_state['research_id'] = ''
    if 'file_id' not in st.session_state:
        st.session_state['file_id'] = ''

    main()