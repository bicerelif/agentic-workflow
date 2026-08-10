from __future__ import annotations

from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents.base_agents import ActionPlanningAgent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / "tests" / ".env")
    openai_api_key = __import__("os").getenv("OPENAI_API_KEY", "")
    agent = ActionPlanningAgent(openai_api_key=openai_api_key, knowledge="Extract steps for simple household tasks.")
    steps = agent.extract_steps_from_prompt("One morning I wanted to have scrambled eggs")
    print("\n".join(steps))


if __name__ == "__main__":
    main()