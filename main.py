from safety_agent import safety_agent
from scenario_agent_with_verificator import scenario_agent_with_verificator

#dane z frontu
user_prompt, scenarios = ("prompt", [("scenario 1", 50), ("scenario 2", 10)])

# tu powinna nastąpić normalizacja wag? tj. skala od 1-100 i normalizacja do 0-1 (float)?

for s in scenarios:
  # unpack obj
  scenario, weight = s

  # PLLUM agent \/
  json_ultra_max_4k_prompt, preferred_domains = scenario_agent_with_verificator(user_prompt, scenario, weight)

  # PLLUM agent \/
  sanitized_user_prompt, sanitized_scenario = safety_agent(json_ultra_max_4k_prompt)

  external_results = []
  # External Agents \/
  for subject in subjects_to_check:
    raw_content1, summary1, url1 = web_search_agent(sanitized_scenario, sanitized_user_prompt, subject, countries_to_check, preferred_domains)
    raw_content2, summary2, url2 = web_search_agent(sanitized_scenario, sanitized_user_prompt, subject, countries_to_check, preferred_domains)

    #throw error if comparison fails
    try:
      raw_content, summary, url = compare_and_verify_web_result(sanitized_scenario, sanitized_user_prompt, subject, countries_to_check, preferred_domains, raw_content1, summary1, url1, raw_content2, summary2, url2)
    except:
      continue

    external_results.append((raw_content, summary, url, scenario, weight))


  # analiza PLLUM
  analyse(user_prompt, scenarios, external_results)