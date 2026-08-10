from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents import AugmentedPromptAgent


def main() -> None:
    agent = AugmentedPromptAgent()
    result = agent.run(
        "Prepare a status note for the Email Router pilot.",
        context="The TPM wants concise, dependency-aware updates.",
        evidence=["Spec is focused on routing, traceability, and metrics."],
    )
    assert result.agent == "AugmentedPromptAgent"
    assert "Context:" in result.output
    assert "Evidence:" in result.output
    print(result.output)
    print("TEST PASSED: AugmentedPromptAgent")


if __name__ == "__main__":
    main()