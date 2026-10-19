import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "evaluation_dataset.json"
REPORT_FILE = BASE_DIR / "evaluation_report.json"
API_URL = "http://127.0.0.1:8000/ask"
REQUEST_TIMEOUT = 120


def load_dataset():
    with DATASET_FILE.open("r", encoding="utf-8-sig") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON list.")

    required = {"id", "question", "expected_answer", "evaluation_type"}

    for item in dataset:
        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"Question {item.get('id', '?')} missing: {missing}"
            )

    return dataset


def ask_studyai(question):
    payload = json.dumps({"question": question}).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        request, timeout=REQUEST_TIMEOUT
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answer(response):
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        for key in ("answer", "response", "result", "output", "message"):
            value = response.get(key)

            if isinstance(value, str):
                return value

            if isinstance(value, dict):
                for nested in ("text", "content", "answer"):
                    if isinstance(value.get(nested), str):
                        return value[nested]

    return json.dumps(response, ensure_ascii=False)


def extract_sources(response):
    if not isinstance(response, dict):
        return []

    for key in ("sources", "citations", "documents", "retrieved_documents"):
        value = response.get(key)

        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str) and value.strip():
            return [value]

    return []


def keyword_check(answer, keywords):
    if not keywords:
        return None, [], []

    text = answer.casefold()
    matched = [k for k in keywords if str(k).casefold() in text]
    missing = [k for k in keywords if str(k).casefold() not in text]

    total = len(matched) + len(missing)
    coverage = round(100 * len(matched) / total, 2) if total else None

    return coverage, matched, missing


def refusal_check(answer):
    phrases = [
        "not provided",
        "does not provide",
        "not mentioned",
        "cannot determine",
        "can't determine",
        "insufficient information",
        "not available in the document",
        "not specified",
        "unable to determine",
        "no information",
        "cannot answer",
        "can't answer",
    ]
    text = answer.casefold()
    return any(phrase in text for phrase in phrases)


def evaluate_question(item):
    print("\n" + "-" * 60)
    print(f"ID: {item['id']}")
    print(f"Question: {item['question']}")

    result = {
        "id": item["id"],
        "question": item["question"],
        "evaluation_type": item["evaluation_type"],
        "expected_answer": item["expected_answer"],
        "actual_answer": None,
        "keyword_coverage_percent": None,
        "matched_keywords": [],
        "missing_keywords": [],
        "citations_present": False,
        "citation_count": 0,
        "refusal_detected": None,
        "response_time_seconds": None,
        "passed": False,
        "error": None,
    }

    try:
        start = time.perf_counter()
        response = ask_studyai(item["question"])
        elapsed = time.perf_counter() - start

        answer = extract_answer(response)
        sources = extract_sources(response)
        keywords = item.get("expected_keywords", [])

        coverage, matched, missing = keyword_check(answer, keywords)
        refused = refusal_check(answer)

        result.update({
            "actual_answer": answer,
            "keyword_coverage_percent": coverage,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "citations_present": bool(sources),
            "citation_count": len(sources),
            "refusal_detected": refused,
            "response_time_seconds": round(elapsed, 2),
        })

        if item["evaluation_type"] == "answerable":
            result["passed"] = (
                coverage is not None
                and coverage >= 60
                and bool(sources)
            )
        else:
            result["passed"] = refused

        print("Answer:", answer)
        print("Keyword coverage:", coverage)
        print("Sources detected:", len(sources))
        print("Response time:", round(elapsed, 2), "seconds")
        print("Result:", "PASS" if result["passed"] else "FAIL")

    except Exception as error:
        result["error"] = str(error)
        print("ERROR:", error)

    return result


def main():
    print("=" * 60)
    print("STUDYAI EVALUATION")
    print("=" * 60)

    try:
        dataset = load_dataset()
    except Exception as error:
        print("Could not load dataset:", error)
        return

    print("Questions loaded:", len(dataset))
    print("API:", API_URL)
    print("\nEnsure the StudyAI server is running in another terminal.")

    results = []

    for index, item in enumerate(dataset, start=1):
        print(f"\nProcessing {index}/{len(dataset)}")
        results.append(evaluate_question(item))
        time.sleep(0.3)

    total = len(results)
    passed = sum(r["passed"] for r in results)
    errors = sum(r["error"] is not None for r in results)
    failed = total - passed - errors

    answerable = [
        r for r in results
        if r["evaluation_type"] == "answerable"
        and r["error"] is None
        and r["keyword_coverage_percent"] is not None
    ]

    coverages = [
        r["keyword_coverage_percent"] for r in answerable
    ]

    times = [
        r["response_time_seconds"] for r in results
        if r["response_time_seconds"] is not None
    ]

    unanswerable = [
        r for r in results
        if r["evaluation_type"] == "unanswerable"
        and r["error"] is None
    ]

    summary = {
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate_percent": round(100 * passed / total, 2) if total else 0,
        "average_keyword_coverage_percent": (
            round(sum(coverages) / len(coverages), 2)
            if coverages else None
        ),
        "average_response_time_seconds": (
            round(sum(times) / len(times), 2) if times else None
        ),
        "unanswerable_questions": len(unanswerable),
        "unanswerable_refusals_detected": sum(
            r["refusal_detected"] is True for r in unanswerable
        ),
    }

    report = {
        "project": "StudyAI",
        "evaluation_time": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "methodology": (
            "Keyword coverage and refusal phrase checks are heuristics, "
            "not proof of semantic or factual accuracy. Citation checking "
            "only verifies that a source field is present."
        ),
        "results": results,
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDetailed report saved to:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
