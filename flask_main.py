import uuid
import os
import pdfplumber
import threading
import time
import json
import logging

from flask import Flask, request, jsonify

# from main import run_engine
from safety_agent import safety_agent
from external_research_agent_2 import ExternalResearchAgent
from predictive_impact_agent import PredictiveImpactAgent
from summary_brief_agent import SummaryBriefAgent
from summary_report_agent import SummaryReportAgent
from scenario_agent_with_verificator import scenario_agent_with_verificator


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



all_external_results_per_scenario = []
HOME_COUNTRY_NAME = "Atlantis"


def run_engine(scenarios_raw, textfiles):
    """
    Główna pętla analityczna, która uruchamia agentów i generuje raporty.
    """
    external_agent = ExternalResearchAgent()
    predictive_agent = PredictiveImpactAgent()
    summary_agent = SummaryReportAgent()
    brief_agent = SummaryBriefAgent()

    # Uwaga: Scenariusze przychodzą jako lista tupli [(scenariusz, waga)]

    all_external_results_per_scenario = []

    for scenario, weight in scenarios_raw:

        logging.info(f"Rozpoczynanie analizy scenariusza (waga={weight}): {scenario[:50]}...")

        # Wyodrębnienie krajów i tematów
        resp = scenario_agent_with_verificator(user_prompt, scenario, weight)

        if isinstance(resp, dict) and "countries" in resp and "subjects" in resp:
            countries = resp["countries"]
            subjects = resp["subjects"]
        else:
            logging.error(f"❌ BŁĘDNA STRUKTURA z scenario_agent dla: {scenario[:30]}")
            continue

        # Safety Agent (Sanityzacja danych)
        sanitized_user_prompt, sanitized_scenario = safety_agent(user_prompt, scenario)

        # External Analysis
        external_results = external_agent.analyze_matrix_for_scenario(
            home_country_name=HOME_COUNTRY_NAME,
            home_context=sanitized_user_prompt,
            scenario=sanitized_scenario,
            foreign_countries=countries,
            subjects=subjects,
        )

        # Predykcja 12 / 36 miesięcy
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

    # --- ZAPISYWANIE WYNIKÓW KOŃCOWYCH ---

    # Raport zbiorczy
    final_report = summary_agent.build_global_report(
        home_context=user_prompt,
        scenarios_data=all_external_results_per_scenario,
    )

    # Krótkie streszczenie
    brief_summary = brief_agent.build_brief_summary(final_report)

    # Zwracamy wszystkie dane do zapisania w job['result']
    return {
        "status": "completed",
        "raw_data": all_external_results_per_scenario,
        "final_report": final_report,
        "brief_summary": brief_summary,
        "context_files_used": textfiles,
    }


# =======================================================
# === LOGIKA ZARZĄDZANIA ZADANIAMI (Worker Functions) ===
# =======================================================

def pass_research_request(research_id, scenarios, textfiles):
    def worker():
        # Ustawienie statusu
        research_queue[research_id]['status'] = 'running'
        try:
            # Uruchomienie głównej pętli analitycznej
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

app = Flask(__name__)

research_queue = {}

# Directory to store uploaded context files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route('/status', methods=['GET'])
def get_status():
    research_id = request.args.get('research_id')
    if not research_id:
        return jsonify({'error': 'Missing research_id query parameter'}), 400
    job = research_queue.get(research_id)
    if not job:
        return jsonify({'error': 'Research id not found'}), 404
    return jsonify({'research_id': research_id, 'status': job.get('status'), 'result': job['result']}), 200


@app.route('/research', methods=['POST'])
def start_research():
    # Parse JSON body
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Missing or invalid JSON body'}), 400

    scenarios = data.get('scenarios')

    context_files = data.get('context_files')

    if context_files is None:
        context_files = []

    textfiles = []
    for file in context_files:
        # txt for now?
        with open(os.path.join(UPLOAD_DIR, file + ".txt"), "r") as f:
            textfiles.append(f.read())

    # Create research job and enqueue
    research_id = uuid.uuid4().hex
    job = {
        'id': research_id,
        'status': 'queued',
        'result': ''
    }
    research_queue[research_id] = job

    pass_research_request(research_id, scenarios, textfiles)
    # Return created id
    return jsonify({'research_id': research_id, 'status': 'queued'}), 202


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Accepts a multipart/form-data file under key 'file', saves it under a random UUID filename
    and returns the id (the filename) as the file identifier.
    """
    if 'file' not in request.files:
        return jsonify(
            {'error': 'No file part in the request (expected key "file")'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file.filename.split(".")[-1] not in ['pdf', 'txt']:
        return jsonify({'error': 'No selected file'}), 400

    # Generate random id and save the file without the original filename to avoid collisions
    file_id = uuid.uuid4().hex
    new_filename = file_id + "." + file.filename.split(".")[-1]
    save_path = os.path.join(UPLOAD_DIR, new_filename)
    file.save(save_path)

    if file.filename.split(".")[-1] == "pdf":
        with pdfplumber.open(file.stream) as pdf, open(
                os.path.join(UPLOAD_DIR, file_id + ".txt"), "w",
                encoding="utf-8") as f:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    f.write(t + '\n')

    return jsonify({'file_id': file_id}), 201



def pass_research_request(research_id, scenarios, textfiles):
    # run long task in pass_research_request_to_engine
    # when the long task is done run: lambda res: set_to_done(research_id)
    def worker():
        # mark job as running
        research_queue[research_id]['status'] = 'running'
        try:
            res = pass_research_request_to_engine(scenarios, textfiles)
            # callback once done
            set_to_done(research_id, res)
        except Exception as e:
            # mark error
            research_queue[research_id]['status'] = 'error'
            research_queue[research_id]['error'] = str(e)
    t = threading.Thread(target=worker, daemon=True)
    t.start()



def pass_research_request_to_engine(scenarios, textfiles):
    eng_resp = run_engine(scenarios)
    return eng_resp



def set_to_done(research_id, result=None):
    # safely update status in dict and attach optional result
    job = research_queue.get(research_id)
    if job is not None:
        job['status'] = 'done'
        if result is not None:
            job['result'] = result


if __name__ == "__main__":
    app.run()
