from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents import DirectPromptAgent


def main() -> None:
    agent = DirectPromptAgent()
    result = agent.run("Draft a TPM status update for the Email Router pilot.")
    assert result.agent == "DirectPromptAgent"
    assert "Direct response" in result.output
    print(result.output)
    print("TEST PASSED: DirectPromptAgent")


if __name__ == "__main__":
    main()