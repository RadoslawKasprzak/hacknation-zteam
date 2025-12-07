import threading
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import config2
from specialized_validator import validator_specialized_agent
import json
import re
import time

# LLM setup
api_key = config2.OPENAI_API_KEY
model_name = "gpt-4.1"

llm = ChatOpenAI(
    model=model_name,
    api_key=lambda: api_key,
    temperature=0.7,
    max_tokens=20000
)


def sanitize_filename(name: str) -> str:
    """Remove unsafe characters for filenames"""
    return re.sub(r'[^A-Za-z0-9_-]', '_', name)


def agent_worker(agent_id, country, field, sanitized_context_str):
    print(f"[Agent-{agent_id}] Launching specialized analysis for field '{field}'...")

    # Prepare the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        template=(
            "You are a specialized research agent.\n"
            "You are given a scenario analysis string (safety agent output):\n"
            "{sanitized_context_str}\n\n"
            "Task:\n"
            "Analyze the given field '{field}' for country '{country}'.\n"
            "Return validated results as a JSON array of objects with keys: area_name, country_analysis, source_name, source_url."
        )
    )

    prompt = prompt_template.format(
        sanitized_context_str=sanitized_context_str,
        country=country,
        field=field
    )

    # --- Synchronous LLM call ---
    response = llm.generate([{"role": "user", "content": prompt}])
    # Depending on langchain version, you might need response.generations[0][0].text
    content = response.generations[0][0].text if hasattr(response, "generations") else response.content
    if not content.strip():
        print(f"[Agent-{agent_id}] Warning: LLM returned empty content for field '{field}'")
        content = "[]"

    # Validate results
    validated_research = validator_specialized_agent(
        sanitized_context_str,
        content
    )

    # Ensure /expert folder exists (absolute path)
    os.makedirs(os.path.join(os.getcwd(), "expert"), exist_ok=True)

    # Save validated results
    safe_country = sanitize_filename(country)
    safe_field = sanitize_filename(field)
    file_path = os.path.join(os.getcwd(), "expert", f"{safe_country}_{safe_field}_{agent_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(validated_research)

    print(f"[Agent-{agent_id}] Saved validated research to {file_path}")


def specialized_agents(sanitized_context_str, country, fields):
    """Launch one thread per field for parallel specialized agents"""
    threads = []

    for i, field in enumerate(fields, start=1):
        t = threading.Thread(target=agent_worker, args=(i, country, field, sanitized_context_str))
        threads.append(t)
        t.start()
        time.sleep(0.2)  # slight stagger to avoid flooding LLM

    # Wait for all threads to finish
    for t in threads:
        t.join()
