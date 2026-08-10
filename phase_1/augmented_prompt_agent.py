from __future__ import annotations

from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents.base_agents import AugmentedPromptAgent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / "tests" / ".env")
    openai_api_key = __import__("os").getenv("OPENAI_API_KEY", "")
    agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona="You are a concise project status assistant.")
    augmented_agent_response = agent.respond("Describe the Email Router pilot in one paragraph.")
    print("The agent likely used general LLM knowledge filtered through the provided persona.")
    print("Specifying the persona narrows tone, focus, and style in the final output.")
    print(augmented_agent_response)


if __name__ == "__main__":
    main()