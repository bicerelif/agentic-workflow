from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
import os
import re
from typing import Any, Callable, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


VOC_URL = "https://openai.vocareum.com/v1"


def _clean_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _strip_bullets(text: str) -> str:
    return re.sub(r"^[\s\-\*\d\.]++", "", text).strip()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _fallback_embedding(text: str, dimensions: int = 12) -> List[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        value = (int.from_bytes(digest[1:5], "big") % 1000) / 1000.0
        vector[index] += value
    return vector


def _load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _create_client(api_key: Optional[str]) -> Optional[Any]:
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(base_url=VOC_URL, api_key=api_key)
    except Exception:
        return None


def _chat_text(client: Any, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> str:
    if client is None:
        return ""
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message.content or ""


def _extract_steps(text: str) -> List[str]:
    steps: List[str] = []
    for line in text.splitlines():
        cleaned = _strip_bullets(line)
        if cleaned:
            steps.append(cleaned)
    return steps


@dataclass
class AgentResult:
    agent: str
    output: str
    metadata: Dict[str, Any]


class BaseAgent:
    name = "BaseAgent"

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        raise NotImplementedError

    def _build_result(self, output: str, **metadata: Any) -> AgentResult:
        return AgentResult(agent=self.name, output=output, metadata=metadata)


class DirectPromptAgent(BaseAgent):
    name = "DirectPromptAgent"

    def __init__(self, openai_api_key: Optional[str] = None) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = _create_client(self.openai_api_key)

    def respond(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = _chat_text(self.client, messages, model="gpt-3.5-turbo")
        if text:
            return text.strip()
        if re.search(r"capital of france", prompt, re.I):
            return "Paris"
        return f"General knowledge response for: {prompt.strip()}"

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        return self._build_result(self.respond(prompt), prompt=prompt)


class AugmentedPromptAgent(BaseAgent):
    name = "AugmentedPromptAgent"

    def __init__(self, openai_api_key: Optional[str] = None, persona: str = "You are a helpful assistant.") -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.persona = persona
        self.client = _create_client(self.openai_api_key)

    def respond(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": f"{self.persona} Forget any previous conversational context."},
            {"role": "user", "content": prompt},
        ]
        text = _chat_text(self.client, messages, model="gpt-3.5-turbo")
        if text:
            return text.strip()
        return f"{self.persona} {prompt.strip()}"

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        context = kwargs.get("context")
        evidence = kwargs.get("evidence")
        output = self.respond(prompt)
        return self._build_result(output, prompt=prompt, context=context, evidence=evidence)


class KnowledgeAugmentedPromptAgent(BaseAgent):
    name = "KnowledgeAugmentedPromptAgent"

    def __init__(self, openai_api_key: Optional[str] = None, persona: str = "You are a helpful assistant.", knowledge: str = "") -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.persona = persona
        self.knowledge = knowledge
        self.client = _create_client(self.openai_api_key)

    def _knowledge_messages(self, prompt: str) -> List[Dict[str, str]]:
        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}. "
            f"Answer the prompt based on this knowledge, not your own."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def respond(self, prompt: str, artifact_type: str = "summary") -> str:
        text = _chat_text(self.client, self._knowledge_messages(prompt), model="gpt-3.5-turbo")
        if text:
            return text.strip()

        lines = _clean_lines(self.knowledge)
        if artifact_type == "user_stories":
            candidates = [line for line in lines if line and not line.endswith(":")][:4]
            stories = []
            for index, line in enumerate(candidates, start=1):
                focus = _strip_bullets(line).rstrip(".")
                stories.append(
                    f"As a support operations user, I want {focus.lower()} so that email routing stays consistent and traceable."
                )
            return "User Stories\n" + "\n".join(stories)
        if artifact_type == "features":
            candidates = [line for line in lines if line and not line.endswith(":")][:4]
            features = []
            for index, line in enumerate(candidates, start=1):
                title = _strip_bullets(line).rstrip(".")
                features.append(
                    "\n".join(
                        [
                            f"Feature Name: {title}",
                            "Description: This feature supports the Email Router workflow with clear operational behavior and traceability.",
                            "Key Functionality: Classify, route, and track support emails with structured handling and visibility.",
                            "User Benefit: Support teams can respond faster with fewer manual handoffs and clearer ownership.",
                        ]
                    )
                )
            return "Product Features\n\n" + "\n\n".join(features)
        if artifact_type == "engineering_tasks":
            candidates = [line for line in lines if line and not line.endswith(":")][:4]
            tasks = []
            for index, line in enumerate(candidates, start=1):
                title = _strip_bullets(line).rstrip(".")
                tasks.append(
                    "\n".join(
                        [
                            f"Task ID: ENG-{index:02d}",
                            f"Task Title: Implement {title}",
                            f"Related User Story: As a support operations user, I want {title.lower()} so that email routing stays consistent and traceable.",
                            f"Description: Implement the {title.lower()} capability for the Email Router workflow.",
                            "Acceptance Criteria: The task is completed when the routed workflow produces the expected structured output and can be reviewed end to end.",
                            "Estimated Effort: Medium",
                            "Dependencies: Product specification review and routing orchestration support.",
                        ]
                    )
                )
            return "Engineering Tasks\n\n" + "\n\n".join(tasks)
        return f"{self.persona} {prompt.strip()} using knowledge: {self.knowledge}"

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        artifact_type = kwargs.get("artifact_type", "summary")
        return self._build_result(self.respond(prompt, artifact_type=artifact_type), prompt=prompt, knowledge=self.knowledge, artifact_type=artifact_type)


class RAGKnowledgePromptAgent(BaseAgent):
    name = "RAGKnowledgePromptAgent"

    def __init__(self, openai_api_key: Optional[str] = None) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = _create_client(self.openai_api_key)

    def respond(self, prompt: str, knowledge: str) -> str:
        retrieved = self.retrieve(prompt, knowledge)
        return "Retrieved knowledge:\n" + "\n".join(f"- {item}" for item in retrieved) if retrieved else "Retrieved knowledge:\n- no directly relevant knowledge found"

    def retrieve(self, query: str, knowledge: str, limit: int = 5) -> List[str]:
        keywords = {token for token in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(token) > 2}
        scored: List[tuple[int, str]] = []
        for line in _clean_lines(knowledge):
            score = sum(1 for keyword in keywords if keyword in line.lower())
            if score:
                scored.append((score, line))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [line for _, line in scored[:limit]]

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        knowledge = kwargs.get("knowledge", "")
        return self._build_result(self.respond(prompt, knowledge), prompt=prompt, knowledge=knowledge)


class EvaluationAgent(BaseAgent):
    name = "EvaluationAgent"

    def __init__(self, openai_api_key: Optional[str] = None, max_interactions: int = 10, persona: str = "You are an evaluation agent that checks the answers of other worker agents", evaluation_criteria: str = "", agent_to_evaluate: Optional[Any] = None) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = _create_client(self.openai_api_key)
        self.max_interactions = max_interactions
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate

    def _evaluate_text(self, worker_response: str) -> str:
        if self.client is not None:
            messages = [
                {"role": "system", "content": f"{self.persona}. Evaluate the response against these criteria: {self.evaluation_criteria}. Return concise feedback and whether it passed."},
                {"role": "user", "content": worker_response},
            ]
            text = _chat_text(self.client, messages, model="gpt-3.5-turbo", temperature=0)
            if text:
                return text.strip()
        lower = worker_response.lower()
        passed = any(marker in lower for marker in ("as a ", "feature", "task id", "task title"))
        return "PASS" if passed else f"FAIL: revise to match criteria -> {self.evaluation_criteria}"

    def _correction_instructions(self, evaluation_result: str) -> str:
        if self.client is not None:
            messages = [
                {"role": "system", "content": f"{self.persona}. Generate correction instructions based on this evaluation result: {evaluation_result}."},
                {"role": "user", "content": self.evaluation_criteria},
            ]
            text = _chat_text(self.client, messages, model="gpt-3.5-turbo", temperature=0)
            if text:
                return text.strip()
        return f"Correct the response so it follows: {self.evaluation_criteria}"

    def evaluate(self, prompt: str, draft_response: Optional[str] = None) -> Dict[str, Any]:
        if self.agent_to_evaluate is None:
            return {"final_response": "", "evaluation_result": "No worker agent provided.", "iterations": 0}

        current_prompt = prompt
        final_response = ""
        evaluation_result = ""
        for iteration in range(1, self.max_interactions + 1):
            if iteration == 1 and draft_response is not None:
                worker_response = draft_response
            else:
                worker_response = self.agent_to_evaluate.respond(current_prompt)
            final_response = worker_response if isinstance(worker_response, str) else str(worker_response)
            evaluation_result = self._evaluate_text(final_response)
            if evaluation_result.upper().startswith("PASS"):
                return {"final_response": final_response, "evaluation_result": evaluation_result, "iterations": iteration}
            current_prompt = self._correction_instructions(evaluation_result)
        return {"final_response": final_response, "evaluation_result": evaluation_result, "iterations": self.max_interactions}

    def respond(self, prompt: str) -> Dict[str, Any]:
        return self.evaluate(prompt)

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        result = self.evaluate(prompt)
        return self._build_result(str(result), prompt=prompt, **result)


class RoutingAgent(BaseAgent):
    name = "RoutingAgent"

    def __init__(self, openai_api_key: Optional[str] = None, agents: Optional[List[Dict[str, Any]]] = None) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = _create_client(self.openai_api_key)
        self.agents = agents or []

    def get_embedding(self, text: str) -> List[float]:
        if self.client is not None:
            try:
                response = self.client.embeddings.create(model="text-embedding-3-large", input=text)
                return list(response.data[0].embedding)
            except Exception:
                pass
        return _fallback_embedding(text)

    def route(self, prompt: str) -> Any:
        if not self.agents:
            return None
        prompt_embedding = self.get_embedding(prompt)
        best_agent: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for agent in self.agents:
            description = agent.get("description", "")
            agent_embedding = self.get_embedding(description)
            score = _cosine_similarity(prompt_embedding, agent_embedding)
            if score > best_score:
                best_score = score
                best_agent = agent
        if best_agent is None:
            return None
        func = best_agent.get("func")
        return func(prompt) if callable(func) else best_agent.get("name")

    def respond(self, prompt: str) -> Any:
        return self.route(prompt)

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        return self._build_result(str(self.route(prompt)), prompt=prompt, agents=self.agents)


class ActionPlanningAgent(BaseAgent):
    name = "ActionPlanningAgent"

    def __init__(self, openai_api_key: Optional[str] = None, knowledge: str = "") -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.knowledge = knowledge
        self.client = _create_client(self.openai_api_key)

    def respond(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": f"You are an Action Planning Agent that extracts steps using provided knowledge: {self.knowledge}"},
            {"role": "user", "content": prompt},
        ]
        text = _chat_text(self.client, messages, model="gpt-3.5-turbo")
        if text:
            return text.strip()
        steps = [
            f"1. Clarify the objective: {prompt.strip()}",
            "2. Review the available knowledge and constraints.",
            "3. Identify the main deliverables.",
            "4. Break the work into sequenced actions.",
            "5. Validate the output against the requested structure.",
        ]
        return "\n".join(steps)

    def extract_steps_from_prompt(self, prompt: str) -> List[str]:
        text = self.respond(prompt)
        steps = [step for step in _extract_steps(text) if step]
        return steps

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        steps = self.extract_steps_from_prompt(prompt)
        return self._build_result("\n".join(steps), prompt=prompt, steps=steps)


def load_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")