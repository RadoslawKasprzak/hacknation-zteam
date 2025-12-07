import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import config2

# LLM setup
api_key = config2.OPENAI_API_KEY
model_name = "gpt-4.1"

llm = ChatOpenAI(
    model=model_name,
    api_key=lambda: api_key,
    temperature=0.7,
    max_tokens=20000
)

def validator_specialized_agent(original_prompt, output_json):
    """
    Validate and possibly correct search results from a specialized agent
    by prompting OpenAI GPT-4.1 via LangChain.
    """
    try:
        research = json.loads(output_json)
    except json.JSONDecodeError:
        research = []

    # Build a chat prompt
    prompt_template = ChatPromptTemplate.from_template(
        template=(
            "You are an expert data validator.\n"
            "Given a user prompt specifying a country and field, "
            "verify the following search results JSON.\n"
            "If the results make sense, return them as-is.\n"
            "If something is inconsistent, correct it while keeping as much valid information as possible.\n"
            "Always return a valid JSON array of objects.\n\n"
            "Original prompt: {original_prompt}\n"
            "Search results: {search_results}"
        )
    )

    # Format the prompt
    prompt = prompt_template.format(
        original_prompt=json.dumps(original_prompt, ensure_ascii=False),
        search_results=json.dumps(research, ensure_ascii=False)
    )

    # Send to LLM
    response = llm(prompt)

    # Parse LLM output
    try:
        validated = json.loads(response.content)
    except json.JSONDecodeError:
        validated = research  # fallback

    return json.dumps(validated, ensure_ascii=False, indent=2)
