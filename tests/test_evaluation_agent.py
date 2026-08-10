from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents import EvaluationAgent


def main() -> None:
    agent = EvaluationAgent()
    result = agent.run(
        "Evaluate user stories for the Email Router pilot.",
        items=[
            "As a platform user, I want emails classified by intent so that support routing is consistent.",
            "As a support lead, I want priority applied to incoming messages so that SLA risk is visible.",
        ],
        artifact_type="user_stories",
    )
    assert result.agent == "EvaluationAgent"
    assert "Evaluated 2 user_stories items." in result.output
    print(result.output)
    print("TEST PASSED: EvaluationAgent")


if __name__ == "__main__":
    main()