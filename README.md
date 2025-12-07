# hacknation-zteam

first install requirements:
`pip install -r requirements.txt`

to run flask app do:
`python flask_main.py`

GET /status?research_id=

POST http://localhost:5000/research
Content-Type: application/json

{
  "scenarios": [{"weight":  10, "text":  "scenario text"}],
  "context_files": ["1af2b32ebca64b24976e7256cffb9bd5"]
}

resp:
{
  "research_id": "xxxx"
}

POST http://127.0.0.1:5000/upload
Content-Type: multipart/form-data

file: <binary>
