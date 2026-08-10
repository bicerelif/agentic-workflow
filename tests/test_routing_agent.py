from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_agents import RoutingAgent


def main() -> None:
    agent = RoutingAgent()
    route = agent.route("Generate user stories from the product spec.")
    assert route == "product_management"
    result = agent.run("Create engineering tasks for implementation.")
    assert result.metadata["route"] == "development_engineering"
    print(f"Route for stories: {route}")
    print(result.output)
    print("TEST PASSED: RoutingAgent")


if __name__ == "__main__":
    main()