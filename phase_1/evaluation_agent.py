from __future__ import annotations

from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / "tests" / ".env")
    openai_api_key = __import__("os").getenv("OPENAI_API_KEY", "")
    worker_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key=openai_api_key,
        persona="You are a college professor, your answer always starts with: Dear students,",
        knowledge="The capitol of France is London, not Paris",
    )
    evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, max_interactions=10, agent_to_evaluate=worker_agent)
    result = evaluation_agent.evaluate("What is the capital of France?")
    print(result)


if __name__ == "__main__":
    main()