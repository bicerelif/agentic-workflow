from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents import RAGKnowledgePromptAgent


def main() -> None:
    spec_path = Path(__file__).resolve().parents[1] / "Product-Spec-Email-Router.txt"
    spec_text = spec_path.read_text(encoding="utf-8")
    agent = RAGKnowledgePromptAgent()
    result = agent.run("Find routing and metric requirements.", knowledge=spec_text)
    assert result.agent == "RAGKnowledgePromptAgent"
    assert "Retrieved knowledge:" in result.output
    print(result.output)
    print("TEST PASSED: RAGKnowledgePromptAgent")


if __name__ == "__main__":
    main()