import argparse
import json
from pathlib import Path
from typing import Any, Optional

from app.agent import Agent


DEFAULT_QUESTIONS_PATH = "data/eval/questions.json"
DEFAULT_OUTPUT_PATH = "data/eval/results/latest.json"


def load_questions(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_judge_messages(
    question: dict[str, Any],
    answer_result: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "id": question["id"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "aliases": question.get("aliases", []),
        "expected_pages": question.get("expected_pages", []),
        "type": question.get("type", ""),
        "actual_answer": answer_result["answer"],
        "actual_citations": answer_result.get("citations", []),
    }
    return [
        {
            "role": "system",
            "content": (
                "你是一个财报问答评测助手。"
                "你只能根据提供的标准答案、别名、期望页码、模型回答和实际页码进行判断。"
                "不要补充外部知识。"
                "请仅返回 JSON，格式为 "
                '{"pass": true, "score": 1.0, "reason": "简短中文理由"}。'
                "score 取值范围为 0 到 1。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def parse_judge_result(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = [line.strip() for line in cleaned.splitlines()]
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    result = json.loads(cleaned)
    return {
        "pass": bool(result["pass"]),
        "score": float(result["score"]),
        "reason": str(result["reason"]),
    }


def evaluate_question(
    agent: Agent,
    question: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    answer_result = agent.ask(question["question"], top_k=top_k)
    judge_response = agent.client.chat.completions.create(
        model=agent.chat_model,
        messages=build_judge_messages(question, answer_result),
        response_format={"type": "json_object"},
    )
    judge_content = judge_response.choices[0].message.content or ""
    judge_result = parse_judge_result(judge_content)
    return {
        "id": question["id"],
        "question": question["question"],
        "type": question.get("type", ""),
        "expected_answer": question["expected_answer"],
        "aliases": question.get("aliases", []),
        "expected_pages": question.get("expected_pages", []),
        "answer": answer_result["answer"],
        "citations": answer_result.get("citations", []),
        "judge": judge_result,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result["judge"]["pass"])
    average_score = (
        sum(result["judge"]["score"] for result in results) / total if total else 0.0
    )
    failed_questions = [
        {
            "id": result["id"],
            "question": result["question"],
            "answer": result["answer"],
            "expected_answer": result["expected_answer"],
            "citations": result["citations"],
            "reason": result["judge"]["reason"],
            "score": result["judge"]["score"],
        }
        for result in results
        if not result["judge"]["pass"]
    ]

    by_type: dict[str, dict[str, Any]] = {}
    for result in results:
        question_type = result["type"] or "unknown"
        type_summary = by_type.setdefault(
            question_type,
            {"total": 0, "passed": 0, "average_score": 0.0},
        )
        type_summary["total"] += 1
        if result["judge"]["pass"]:
            type_summary["passed"] += 1
        type_summary["average_score"] += result["judge"]["score"]

    for type_summary in by_type.values():
        type_summary["average_score"] = round(
            type_summary["average_score"] / type_summary["total"],
            4,
        )

    return {
        "total": total,
        "passed": passed,
        "average_score": round(average_score, 4),
        "overall_score": round(average_score, 4),
        "failed_questions": failed_questions,
        "by_type": by_type,
    }


def resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the financial report QA pipeline.",
        add_help=add_help,
    )
    parser.add_argument(
        "--questions-path",
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to the evaluation questions JSON file",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the evaluation results JSON file",
    )
    parser.add_argument(
        "--embeddings-path",
        default="data/processed/embeddings.json",
        help="Path to the embeddings JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many chunks to retrieve for each question",
    )
    return parser


def run_eval_command(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    questions_path = resolve_path(project_root, args.questions_path)
    output_path = resolve_path(project_root, args.output_path)
    questions = load_questions(questions_path)
    results = []

    with Agent.from_env() as agent:
        for question in questions:
            result = evaluate_question(agent, question, top_k=args.top_k)
            results.append(result)
            judge = result["judge"]
            status = "PASS" if judge["pass"] else "FAIL"
            print(f"[{status}] {result['id']} {result['question']}")

    summary = summarize_results(results)
    output = {"summary": summary, "results": results}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Pass rate: {summary['passed']}/{summary['total']}")
    print(f"Overall score: {summary['overall_score']:.4f}")
    print(f"Average score: {summary['average_score']:.4f}")
    if summary["failed_questions"]:
        print("Failed questions:")
        for failed in summary["failed_questions"]:
            print(f"- {failed['id']}: {failed['question']}")
            print(f"  answer: {failed['answer']}")
            print(f"  expected: {failed['expected_answer']}")
            print(f"  citations: {failed['citations']}")
            print(f"  reason: {failed['reason']}")
    print(f"Saved results to {output_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_eval_command(args)


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_QUESTIONS_PATH",
    "build_arg_parser",
    "build_judge_messages",
    "evaluate_question",
    "load_questions",
    "main",
    "parse_judge_result",
    "resolve_path",
    "run_eval_command",
    "summarize_results",
]
