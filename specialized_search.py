import threading
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import config2
from specialized_validator import validator_specialized_agent
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


def extract_country_and_fields(sanitized_context_str):
    """Extracts country and relevant fields from a safety agent string."""
    country_match = re.search(r"Nazwa państwa:\s*(.+)", sanitized_context_str)
    country = country_match.group(1).strip() if country_match else "Unknown"

    field_patterns = [
        r"Istotne cechy położenia geograficznego:\s*(.+)",
        r"Silne strony gospodarki:\s*(.+)",
        r"Potencjalne zagrożenie militarne:\s*(.+)"
    ]

    fields = []
    for pattern in field_patterns:
        match = re.search(pattern, sanitized_context_str, flags=re.DOTALL)
        if match:
            fields.append(match.group(1).strip())

    # fallback to at least one field
    if not fields:
        fields = ["general"]

    return country, fields


def agent_worker(agent_id, country, field, sanitized_context_str):
    print(f"[Agent-{agent_id}] Launching specialized analysis for field '{field}'...")

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

    # Call LLM
    try:
        response = llm.generate([{"role": "user", "content": prompt}])
        # depending on LangChain version
        content = response.generations[0][0].text if hasattr(response, "generations") else response.content
        if not content.strip():
            content = "[]"
            print(f"[Agent-{agent_id}] LLM returned empty content, defaulting to empty array.")
    except Exception as e:
        print(f"[Agent-{agent_id}] Error calling LLM: {e}")
        content = "[]"

    # Validate output
    try:
        validated_research = validator_specialized_agent(sanitized_context_str, content)
        if not validated_research.strip():
            validated_research = "[]"
            print(f"[Agent-{agent_id}] Validator returned empty content, defaulting to empty array.")
    except Exception as e:
        print(f"[Agent-{agent_id}] Error in validator: {e}")
        validated_research = "[]"

    # Ensure /expert exists
    expert_dir = os.path.join(os.getcwd(), "expert")
    os.makedirs(expert_dir, exist_ok=True)

    # Save validated results
    safe_country = sanitize_filename(country)
    safe_field = sanitize_filename(field)
    file_path = os.path.join(expert_dir, f"{safe_country}_{safe_field}_{agent_id}.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(validated_research)
        print(f"[Agent-{agent_id}] Saved validated research to {file_path}")
    except Exception as e:
        print(f"[Agent-{agent_id}] Error saving file: {e}")


def specialized_agents(sanitized_context_str):
    """Extract country and fields, launch one thread per field"""
    print(sanitized_context_str)
    country, fields = extract_country_and_fields(sanitized_context_str)
    threads = []

    for i, field in enumerate(fields, start=1):
        t = threading.Thread(target=agent_worker, args=(i, country, field, sanitized_context_str))
        threads.append(t)
        t.start()
        time.sleep(0.2)  # small stagger to prevent LLM overload

    for t in threads:
        t.join()
