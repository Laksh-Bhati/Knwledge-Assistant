import csv
import sys
from pathlib import Path

from src.config import TOP_K, MIN_SCORE, CHUNK_SIZE, CHUNK_OVERLAP
from src.rag import answer_question, NOT_FOUND
from src.vectorstore import search, count_chunks

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.csv"
RESULTS_FILE = EVAL_DIR / "results.csv"


def load_questions():
    with open(QUESTIONS_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_one(row):
    """Run one test question and check retrieval, answer, citations and refusal."""
    question = row["question"]
    should_answer = row["should_answer"].strip().lower() == "yes"
    expected_page = int(row["expected_page"]) if row["expected_page"].strip() else None
    keywords = [k.strip().lower() for k in row["keywords"].split("|") if k.strip()]

    # Retrieval check: what does the vector database rank first?
    hits = search(question)
    top_page = hits[0]["page"] if hits else None
    top_score = hits[0]["score"] if hits else 0.0

    # Full RAG answer
    result = answer_question(question)
    answer = result["answer"]
    refused = NOT_FOUND in answer

    if should_answer:
        retrieval_ok = top_page == expected_page
        keyword_ok = (not refused) and any(k in answer.lower() for k in keywords)
        citation_ok = "[" in answer and "]" in answer
        passed = retrieval_ok and keyword_ok and citation_ok
    else:
        retrieval_ok = keyword_ok = citation_ok = None  # not applicable
        passed = refused

    return {
        "question": question,
        "should_answer": "yes" if should_answer else "no",
        "expected_page": expected_page or "",
        "top_page": top_page or "",
        "top_score": top_score,
        "retrieval_ok": retrieval_ok,
        "keyword_ok": keyword_ok,
        "citation_ok": citation_ok,
        "refused": refused,
        "used_llm": result["used_llm"],
        "passed": passed,
        "answer": answer,
    }


def main():
    if count_chunks() == 0:
        print("❌ Knowledge base is empty. Ingest the sample document first:")
        print("   python -m src.vectorstore ingest data\\sample_docs\\acme_onboarding_guide.pdf")
        sys.exit(1)

    rows = load_questions()
    print(f"Running {len(rows)} questions "
          f"(chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}, top_k={TOP_K}, min_score={MIN_SCORE})\n")

    results = []
    for row in rows:
        try:
            r = evaluate_one(row)
        except Exception as e:
            print(f"⚠️  ERROR  {row['question']}  ({type(e).__name__})")
            continue
        results.append(r)
        mark = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{mark}  score={r['top_score']:.3f}  page={r['top_page']}  {r['question']}")

    # ----- Summary -----
    answerable = [r for r in results if r["should_answer"] == "yes"]
    unanswerable = [r for r in results if r["should_answer"] == "no"]

    def count(items, key):
        return sum(1 for r in items if r[key])

    print("\n========== SUMMARY ==========")
    print(f"Top-1 retrieval accuracy : {count(answerable, 'retrieval_ok')}/{len(answerable)}")
    print(f"Answer correctness       : {count(answerable, 'keyword_ok')}/{len(answerable)}")
    print(f"Citations present        : {count(answerable, 'citation_ok')}/{len(answerable)}")
    print(f"Correct refusals         : {count(unanswerable, 'refused')}/{len(unanswerable)}")
    print(f"False refusals           : {count(answerable, 'refused')}/{len(answerable)}")
    print(f"Overall passed           : {count(results, 'passed')}/{len(results)}")

    # Save full details (including answers) for manual review
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()