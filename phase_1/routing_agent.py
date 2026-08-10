from __future__ import annotations

from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / "tests" / ".env")
    openai_api_key = __import__("os").getenv("OPENAI_API_KEY", "")
    texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona="Texas specialist", knowledge="Texas facts and history")
    europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona="Europe specialist", knowledge="Europe facts and history")
    math_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona="Math specialist", knowledge="Math and arithmetic facts")

    router = RoutingAgent(openai_api_key=openai_api_key)
    router.agents = [
        {"name": "Texas Agent", "description": "Responsible for Texas-related knowledge and history.", "func": lambda prompt: texas_agent.respond(prompt)},
        {"name": "Europe Agent", "description": "Responsible for Europe-related knowledge and history.", "func": lambda prompt: europe_agent.respond(prompt)},
        {"name": "Math Agent", "description": "Responsible for math-related prompts and calculations.", "func": lambda prompt: math_agent.respond(prompt)},
    ]

    for prompt in [
        "Tell me about the history of Rome, Texas",
        "Tell me about the history of Rome, Italy",
        "One story takes 2 days, and there are 20 stories",
    ]:
        print(router.route(prompt))


if __name__ == "__main__":
    main()