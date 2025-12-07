import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import config2

system_prompt = """Jesteś analitykiem bezpieczeństwa dla fikcyjnego kraju Atlantis. 
W tekście który dostaniesz, usuń TYLKO dane poufne ze scenario i context. 
Wysyłaj spowrotem TYLKO JSON: obiekt z trzema polami: scenario, context oraz summary (podsumowanie usuwania danych poufnych).
"""

user_prompt_template = """scenario:
{scenario}

context:
{context}
"""

api_key = config2.OPENAI_API_KEY
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


def safety_agent(user_prompt, scenario):
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user",
     user_prompt_template)
  ])

  chain = prompt | llm
  response = chain.invoke({"scenario": scenario, "context": user_prompt})
  loads = json.loads(response.content)
  return loads['context'], loads['scenario'], loads['summary']
