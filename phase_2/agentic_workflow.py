from __future__ import annotations

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return None

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "Product-Spec-Email-Router.txt"
SHARED_DOTENV_PATH = PROJECT_ROOT / "phase_1" / "tests" / ".env"


def main() -> None:
    load_dotenv(SHARED_DOTENV_PATH)
    openai_api_key = ""
    product_spec = SPEC_PATH.read_text(encoding="utf-8")

    knowledge_action_planning = "You are an Action Planning Agent that extracts steps using provided knowledge."
    action_planning_agent = ActionPlanningAgent(openai_api_key=openai_api_key, knowledge=knowledge_action_planning)

    knowledge_product_manager = (
        "You are a product manager. Create user stories only. " + product_spec
    )
    persona_product_manager = "You are a product manager who writes user stories."
    product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key=openai_api_key,
        persona=persona_product_manager,
        knowledge=knowledge_product_manager,
    )
    product_manager_evaluation_agent = EvaluationAgent(
        openai_api_key=openai_api_key,
        max_interactions=10,
        persona="You are an evaluation agent that checks the answers of other worker agents",
        evaluation_criteria="The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value].",
        agent_to_evaluate=product_manager_knowledge_agent,
    )

    knowledge_program_manager = "You are a program manager. Create product features only. " + product_spec
    persona_program_manager = "You are a program manager who writes product features."
    program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_program_manager, knowledge=knowledge_program_manager)
    program_manager_evaluation_agent = EvaluationAgent(
        openai_api_key=openai_api_key,
        max_interactions=10,
        persona="You are an evaluation agent that checks the answers of other worker agents",
        evaluation_criteria=(
            "The answer should be product features that follow the following structure: "
            "Feature Name: A clear, concise title that identifies the capability\n"
            "Description: A brief explanation of what the feature does and its purpose\n"
            "Key Functionality: The specific capabilities or actions the feature provides\n"
            "User Benefit: How this feature creates value for the user"
        ),
        agent_to_evaluate=program_manager_knowledge_agent,
    )

    knowledge_dev_engineer = "You are a development engineer. Create tasks only. " + product_spec
    persona_dev_engineer = "You are a development engineer who writes engineering tasks."
    development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_dev_engineer, knowledge=knowledge_dev_engineer)
    development_engineer_evaluation_agent = EvaluationAgent(
        openai_api_key=openai_api_key,
        max_interactions=10,
        persona="You are an evaluation agent that checks the answers of other worker agents",
        evaluation_criteria=(
            "The answer should be tasks following this exact structure: "
            "Task ID: A unique identifier  for tracking purposes\n"
            "Task Title: Brief description of the specific development work\n"
            "Related User Story: Reference to the parent user story\n"
            "Description: Detailed explanation of the technical work required\n"
            "Acceptance Criteria: Specific requirements that must be met for completion\n"
            "Estimated Effort: Time or complexity estimation\n"
            "Dependencies: Any tasks that must be completed first"
        ),
        agent_to_evaluate=development_engineer_knowledge_agent,
    )

    routing_agent = RoutingAgent(openai_api_key=openai_api_key)

    def product_manager_support_function(query: str) -> str:
        draft_response = product_manager_knowledge_agent.respond(query, artifact_type="user_stories")
        evaluation_result = product_manager_evaluation_agent.evaluate(query, draft_response=draft_response)
        return evaluation_result["final_response"]

    def program_manager_support_function(query: str) -> str:
        draft_response = program_manager_knowledge_agent.respond(query, artifact_type="features")
        evaluation_result = program_manager_evaluation_agent.evaluate(query, draft_response=draft_response)
        return evaluation_result["final_response"]

    def development_engineer_support_function(query: str) -> str:
        draft_response = development_engineer_knowledge_agent.respond(query, artifact_type="engineering_tasks")
        evaluation_result = development_engineer_evaluation_agent.evaluate(query, draft_response=draft_response)
        return evaluation_result["final_response"]

    routing_agent.agents = [
        {
            "name": "Product Manager",
            "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories",
            "func": lambda x: product_manager_support_function(x),
        },
        {
            "name": "Program Manager",
            "description": "Responsible for defining product features only. Does not define user stories or tasks.",
            "func": lambda x: program_manager_support_function(x),
        },
        {
            "name": "Development Engineer",
            "description": "Responsible for defining engineering tasks only. Does not define user stories or features.",
            "func": lambda x: development_engineer_support_function(x),
        },
    ]

    workflow_steps = [
        "Generate Email Router user stories using the required user story format.",
        "Generate Email Router product features with Feature Name, Description, Key Functionality, and User Benefit.",
        "Generate Email Router engineering tasks with Task ID, Task Title, Related User Story, Description, Acceptance Criteria, Estimated Effort, and Dependencies.",
    ]
    completed_steps = []
    for step in workflow_steps:
        print(f"Processing step: {step}")
        result = routing_agent.route(step)
        completed_steps.append(result)
        print(f"Step result: {result}")

    final_output = "\n\n".join(str(step) for step in completed_steps if step)
    output_paths = [
        PROJECT_ROOT / "artifacts" / "test-outputs" / "agentic_workflow.txt",
        PROJECT_ROOT / "phase_2" / "artifacts" / "test-outputs" / "agentic_workflow.txt",
    ]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_output, encoding="utf-8")

    print("Final workflow output:")
    print(final_output if final_output else "No completed steps")


if __name__ == "__main__":
    main()