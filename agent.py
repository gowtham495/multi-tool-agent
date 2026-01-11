"""Agent Configuration and Execution Module.
This module initializes a multi-tool agent capable of retrieving weather and time information.
It utilizes the Google ADK framework and LiteLLM to interface with local models via Ollama.

Attributes:
    MODEL (str): The name of the LLM model to use, sourced from the OLLAMA_MODEL
                 environment variable or defaulting to 'qwen2.5:7b-instruct'.
    root_agent (Agent): The primary agent instance configured with weather and
                        time retrieval tools.

"""
import os

from tools import get_weather, get_current_time
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import litellm

# Enable debug logging for litellm
litellm._turn_on_debug()

# Ollama model should be already present in env and serving
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Initialize the root agent
root_agent = Agent(
    model=LiteLlm(model="ollama_chat/" + MODEL), 
    name="weather_time_agent",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction="""
      You are a helpful agent who can answer user questions about the time and weather in a city.
    """,
    tools=[get_weather,get_current_time],
)