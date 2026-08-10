from __future__ import annotations

from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents.base_agents import DirectPromptAgent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / "tests" / ".env")
    openai_api_key = __import__("os").getenv("OPENAI_API_KEY", "")
    direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)
    response = direct_agent.respond("What is the Capital of France?")
    print("Knowledge source: general knowledge from the selected LLM model.")
    print(response)


if __name__ == "__main__":
    main()