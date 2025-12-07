from typing import TypedDict, List
from langchain_core.messages import BaseMessage

from scenario_agent import ScenarioOutput

class AgentState(TypedDict):
    """Reprezentuje stan obiegu pracy agentów (GraphState)."""
    scenario_input: str
    weight: int
    scenario_analysis: ScenarioOutput
    research_results: List[str]
    status: str

class AgentState(TypedDict):
    """
    Reprezentuje stan obiegu pracy agentów (GraphState) dla MSZ.
    """

    scenario_input: str

    scenario_analysis: ScenarioOutput

    research_results: List[str]

    status: str

    weight: int