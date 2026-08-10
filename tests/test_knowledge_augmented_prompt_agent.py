from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents import KnowledgeAugmentedPromptAgent


def main() -> None:
    spec_path = Path(__file__).resolve().parents[1] / "Product-Spec-Email-Router.txt"
    spec_text = spec_path.read_text(encoding="utf-8")
    agent = KnowledgeAugmentedPromptAgent()
    result = agent.run(
        "Generate user stories for the Email Router pilot.",
        knowledge=spec_text,
        artifact_type="user_stories",
    )
    assert result.agent == "KnowledgeAugmentedPromptAgent"
    assert "User Stories" in result.output
    print(result.output)
    print("TEST PASSED: KnowledgeAugmentedPromptAgent")


if __name__ == "__main__":
    main()