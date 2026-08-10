from __future__ import annotations

from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents.base_agents import RAGKnowledgePromptAgent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / "tests" / ".env")
    openai_api_key = __import__("os").getenv("OPENAI_API_KEY", "")
    agent = RAGKnowledgePromptAgent(openai_api_key=openai_api_key)
    knowledge = Path(__file__).resolve().parents[1] / "Product-Spec-Email-Router.txt"
    print(agent.respond("Find routing and metric requirements.", knowledge.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()