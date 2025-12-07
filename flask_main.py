import uuid
import os
import pdfplumber
import threading
import time

from flask import Flask, request, jsonify

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
    return jsonify({'research_id': research_id, 'status': job.get('status')}), 200


@app.route('/research', methods=['POST'])
def start_research():
    # Parse JSON body
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Missing or invalid JSON body'}), 400

    user_prompt = data.get('user_prompt')

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
        'status': 'queued'
    }
    research_queue[research_id] = job

    pass_research_request(research_id, user_prompt, scenarios, textfiles)
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



def pass_research_request(research_id, user_prompt, scenarios, textfiles):
    # run long task in pass_research_request_to_engine
    # when the long task is done run: lambda res: set_to_done(research_id)
    def worker():
        # mark job as running
        research_queue[research_id]['status'] = 'running'
        try:
            res = pass_research_request_to_engine(user_prompt, scenarios, textfiles)
            # callback once done
            set_to_done(research_id, res)
        except Exception as e:
            # mark error
            research_queue[research_id]['status'] = 'error'
            research_queue[research_id]['error'] = str(e)
    t = threading.Thread(target=worker, daemon=True)
    t.start()



def pass_research_request_to_engine(user_prompt, scenarios, textfiles):
    # long task
    # simulate work
    time.sleep(10)
    # simple result aggregation
    return { }



def set_to_done(research_id, result=None):
    # safely update status in dict and attach optional result
    job = research_queue.get(research_id)
    if job is not None:
        job['status'] = 'done'
        if result is not None:
            job['result'] = result


if __name__ == "__main__":
    app.run()
