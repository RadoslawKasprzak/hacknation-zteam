import threading
import time
from urllib.parse import urlparse

# Example whitelist
MSZ_OFFICIAL_SOURCES = [
    # Germany
    "auswaertiges-amt.de", "bmvg.de", "bmi.bund.de", "bmwk.de",
    "bmz.de", "bmuv.de", "bmbf.de", "bmdv.bund.de", "bundesregierung.de",

    # France
    "diplomatie.gouv.fr", "defense.gouv.fr", "interieur.gouv.fr",
    "economie.gouv.fr", "douane.gouv.fr", "ecologie.gouv.fr",
    "enseignementsup-recherche.gouv.fr", "numerique.gouv.fr", "education.gouv.fr",

    # Russia
    "mid.ru/en", "mil.ru", "government.ru/en", "economy.gov.ru",
    "minenergo.gov.ru", "minobrnauki.gov.ru", "edu.gov.ru", "digital.gov.ru",
]


def extract_domain(url):
    """Normalize URL to domain for whitelist checking"""
    parsed = urlparse(url)
    domain = parsed.netloc if parsed.netloc else parsed.path
    # remove www
    return domain.replace("www.", "")


def agent_worker(agent_id, country, field, json_data, whitelist):
    """Worker function for a single specialized agent"""
    print(f"[Agent-{agent_id}] Starting search for country={country}, field={field}...")

    results = []

    for area in json_data.get("global_areas", []):
        # Check if the field matches area_name
        if field.lower() in area["area_name"].lower():
            for analysis in area.get("country_impact_analysis", []):
                if country.lower() in analysis["country"].lower():
                    # Gather information from sources if whitelisted
                    sources = area.get("information_sources", [])
                    for src in sources:
                        domain = extract_domain(src.get("url_or_description", ""))
                        if any(allowed in domain for allowed in whitelist):
                            results.append({
                                "area_name": area["area_name"],
                                "country_analysis": analysis["country"],
                                "source_name": src["source_name"],
                                "source_url": src.get("url_or_description")
                            })

    print(f"[Agent-{agent_id}] Found {len(results)} results.")
    for res in results:
        print(f"[Agent-{agent_id}] {res}")


def specialized_agents(sanitized_user_prompt, whitelist=MSZ_OFFICIAL_SOURCES):
    """
    Launches specialized agents based on user prompt.
    The prompt should contain:
        - country
        - field of expertise
    """
    # Example parsing of user prompt (replace with proper NLP if needed)
    country = sanitized_user_prompt.get("country")
    field = sanitized_user_prompt.get("field")

    # Launch a thread per agent (in this example, only one agent per prompt)
    threads = []
    agent_id = 1
    t = threading.Thread(target=agent_worker, args=(agent_id, country, field, whitelist))
    t.start()
    threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()
