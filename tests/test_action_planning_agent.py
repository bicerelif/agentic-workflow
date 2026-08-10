from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents import ActionPlanningAgent


def main() -> None:
    agent = ActionPlanningAgent()
    result = agent.run("Build the Email Router pilot workflow for technical project management.")
    assert result.agent == "ActionPlanningAgent"
    assert len(result.metadata["steps"]) == 7
    print(result.output)
    print("TEST PASSED: ActionPlanningAgent")


if __name__ == "__main__":
    main()