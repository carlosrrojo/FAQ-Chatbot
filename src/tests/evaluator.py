"""
RAG Evaluation harness.

Evaluates the retrieve → generate pipeline using four metrics:
  - Faithfulness      (answer supported by retrieved context)
  - Answer Relevancy  (answer addresses the question)
  - Context Precision (retrieved chunks are useful)
  - Context Recall    (context covers the ground-truth answer)
"""

from benchmarks.eval_data import FAQ_QUERIES
from src.config import COLLECTION, EMBEDDING_MODEL
from sqlalchemy.orm.collections import collection
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from benchmarks.eval_data import FAQ_QUERIES
from langchain_core.globals import set_debug
from src.rag.agent import retrieve_documents
from src.domain.orchestrator import RAGOrchestrator
from src.domain.models import ChatRequest
from src.infrastructure.memory.sqlite_memory import SqliteMemoryAdapter
from src.infrastructure.embeddings import get_embeddings
import src.rag.agent

set_debug(False)

EVAL_DATA = FAQ_QUERIES


# FR-EVL-01 AC-1: minimum score thresholds (Gemini judge, Spanish partition)
METRIC_THRESHOLDS: dict[str, float] = {
    "faithfulness":     0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall":   0.70,
}

# FR-EVL-01 AC-4 / NFR-PERF: latency thresholds (seconds)
NFR_E2E_THRESHOLD_S: float = 15.0        # NFR-PERF-01: ≤15 s end-to-end
NFR_RETRIEVAL_THRESHOLD_S: float = 3.0    # NFR-PERF-02: ≤3 s retrieval

orchestrator_mem    = RAGOrchestrator(memory_store=SqliteMemoryAdapter())
orchestrator_no_mem = RAGOrchestrator(memory_store=None)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _parse_score(text) -> float:
    if text is None:
        return 0.0
    if not isinstance(text, str):
        text = str(text)
    match = re.search(r"0\.\d+|1\.0|0|1", text)
    return min(max(float(match.group()), 0.0), 1.0) if match else 0.0


def calculate_faithfulness(question: str, answer: str, contexts: list[str], llm) -> float:
    """Check whether the answer is fully supported by the retrieved context."""
    if not answer or not contexts:
        return 0.0
    prompt = (
        "Given the Question, Context, and Answer, evaluate if the Answer is completely "
        "supported by the Context.\n"
        "Return ONLY a float between 0.0 and 1.0. No other text.\n"
        f"Question: {question}\nContext: {chr(10).join(contexts)}\nAnswer: {answer}\nScore:"
    )
    return _parse_score(llm.invoke(prompt).content)


def calculate_answer_relevancy(
    question: str, answer: str, llm, embeddings
) -> float:
    """
    Measure how relevant the answer is to the question by generating synthetic
    questions and comparing their embeddings to the original question.
    """
    if not answer:
        return 0.0

    gen_q_prompt = (
        "Generate 3 different practical questions that this answer could be responding to.\n"
        "Return a JSON list of strings. No other text.\n"
        f"Answer: {answer}\nOutput:"
    )
    res = llm.invoke(gen_q_prompt).content
    if isinstance(res, list):
        text_parts = []
        for part in res:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        res_str = "".join(text_parts)
    else:
        res_str = str(res)

    try:
        cleaned_res = res_str
        if "```json" in cleaned_res:
            cleaned_res = cleaned_res.split("```json")[-1].split("```")[0]
        elif "```" in cleaned_res:
            cleaned_res = cleaned_res.split("```")[-1].split("```")[0]
        gen_questions = json.loads(cleaned_res)
    except Exception:
        gen_questions = [q.strip("- 1234567890.") for q in res_str.split("\n") if q.strip()]

    if not gen_questions or not isinstance(gen_questions, list):
        return 0.0

    gen_questions = [str(q) for q in gen_questions if q]

    if not gen_questions:
        return 0.0

    orig_emb = embeddings.embed_query(question)
    gen_embs = embeddings.embed_documents(gen_questions)
    return float(np.mean(cosine_similarity([orig_emb], gen_embs)[0]))


def calculate_context_recall(
    question: str, ground_truth: str, contexts: list[str], llm
) -> float:
    """Check what fraction of the ground truth is inferable from the context."""
    if not ground_truth or not contexts:
        return 0.0
    prompt = (
        "Given the Context and Ground Truth, evaluate what proportion of the Ground Truth "
        "can be inferred solely from the Context.\n"
        "Return ONLY a float between 0.0 and 1.0. No other text.\n"
        f"Context: {chr(10).join(contexts)}\nGround Truth: {ground_truth}\nScore:"
    )
    return _parse_score(llm.invoke(prompt).content)


def calculate_context_precision(
    question: str, ground_truth: str, retrieved_docs: list, llm
) -> float:
    """Measure how useful the retrieved context is for arriving at the ground truth."""
    if not retrieved_docs or not ground_truth:
        return 0.0
    context_str = "\n".join(doc.page_content for doc in retrieved_docs)
    prompt = (
        "Given the Question, Ground Truth, and Context, evaluate how useful the Context "
        "is for arriving at the Ground Truth.\n"
        "Return ONLY a float between 0.0 and 1.0. No other text.\n"
        f"Question: {question}\nGround Truth: {ground_truth}\nContext: {context_str}\nScore:"
    )
    return _parse_score(llm.invoke(prompt).content)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def run_evaluator(collection: str = "Espazo Nature", evaluator: str = "ollama", limit: int = None) -> dict:
    """
    Run the full evaluation harness.
    Returns the structured report dict (also persisted as JSON to ``evals/``).
    """
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n{'='*44}")
    print(f"Collection: {collection}")
    print(f"Evaluator LLM: {evaluator}")
    print(f"{'='*44}")

    if evaluator == "gemini":
        model      = ChatGoogleGenerativeAI(model="gemini-flash-latest")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        model      = ChatOllama(model="llama3.1")
        #embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        embeddings = get_embeddings()
    
    results: dict[str, dict[str, list[float]]] = {
        "es": {
            "faithfulness":    [],
            "answer_relevancy": [],
            "context_precision":[],
            "context_recall":  [],
        },
        "en": {
            "faithfulness":    [],
            "answer_relevancy": [],
            "context_precision":[],
            "context_recall":  [],
        }
    }
    per_query: list[dict] = []
    retrieval_latencies: list[float] = []   # seconds
    e2e_latencies: list[float] = []         # seconds

    if limit:
        es_in = [item for item in EVAL_DATA if item.get("language") == "es" and not ("fuera del ámbito" in item["ground_truth"] or "outside the scope" in item["ground_truth"])]
        en_in = [item for item in EVAL_DATA if item.get("language") == "en" and not ("fuera del ámbito" in item["ground_truth"] or "outside the scope" in item["ground_truth"])]
        oos = [item for item in EVAL_DATA if "fuera del ámbito" in item["ground_truth"] or "outside the scope" in item["ground_truth"]]
        
        eval_items = []
        from math import ceil
        n_es = min(len(es_in), ceil(limit * 0.4))
        n_en = min(len(en_in), ceil(limit * 0.4))
        n_oos = min(len(oos), limit - n_es - n_en)
        if n_oos < 0:
            n_oos = 0
            
        eval_items.extend(es_in[:n_es])
        eval_items.extend(en_in[:n_en])
        eval_items.extend(oos[:n_oos])
        
        remaining = limit - len(eval_items)
        if remaining > 0:
            for item in EVAL_DATA:
                if item not in eval_items:
                    eval_items.append(item)
                    remaining -= 1
                    if remaining == 0:
                        break
    else:
        eval_items = EVAL_DATA

    for idx, item in enumerate(eval_items):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        lang         = item.get("language", "es")
        stratum      = item.get("stratum", "")

        # ── Timed retrieval (AC-4 / NFR-PERF-02) ─────────────────────
        t_ret_start = time.perf_counter()
        _, artifact = retrieve_documents.func(question)
        retrieval_s = time.perf_counter() - t_ret_start
        retrieval_latencies.append(retrieval_s)

        retrieved_docs = artifact["docs"]
        contexts = [doc.page_content for doc in retrieved_docs]

        # ── Timed end-to-end (AC-4 / NFR-PERF-01) ────────────────────
        use_memory = item.get("memory", False)
        orch = orchestrator_mem if use_memory else orchestrator_no_mem
        t_e2e_start = time.perf_counter()
        answer = orch.generate_reply(
            ChatRequest(platform="eval", text=question, sender_id="eval-runner")
        ).text
        e2e_s = time.perf_counter() - t_e2e_start
        e2e_latencies.append(e2e_s)

        if not ground_truth:
            print("  ⚠ No ground truth — skipping metrics.")
            per_query.append({"question": question, "language": lang, "skipped": True})
            continue

        f_score  = calculate_faithfulness(question, answer, contexts, model)
        ar_score = calculate_answer_relevancy(question, answer, model, embeddings)
        cp_score = calculate_context_precision(question, ground_truth, retrieved_docs, model)
        cr_score = calculate_context_recall(question, ground_truth, contexts, model)

        # Throttle API calls to avoid Gemini rate limits
        if evaluator == "gemini":
            time.sleep(5)

        print(f"\nQ{idx+1}/{len(EVAL_DATA)}: {question} ({lang.upper()}) |F: {f_score:.2f} | AR: {ar_score:.2f} | CP: {cp_score:.2f} | CR: {cr_score:.2f}")
        print("=======================================================")
        print("\t\t\t\t\tAnswer:\n")
        print("=======================================================")
        print(answer)
        print("=======================================================")
        print("\t\t\t\t\tContexts:\n")
        print("=======================================================")
        short_contexts = [f"{c[:80]}..." if len(c) > 80 else c for c in contexts]
        print(f"Contexts:\n {short_contexts}")
        print("=======================================================")
        
        if lang not in results:
            results[lang] = {m: [] for m in METRIC_THRESHOLDS}
            
        results[lang]["faithfulness"].append(f_score)
        results[lang]["answer_relevancy"].append(ar_score)
        results[lang]["context_precision"].append(cp_score)
        results[lang]["context_recall"].append(cr_score)

        per_query.append({
            "question": question,
            "language": lang,
            "stratum": stratum,
            "faithfulness": round(f_score, 4),
            "answer_relevancy": round(ar_score, 4),
            "context_precision": round(cp_score, 4),
            "context_recall": round(cr_score, 4),
            "retrieval_latency_s": round(retrieval_s, 3),
            "e2e_latency_s": round(e2e_s, 3),
        })
    # ------------------------------------------------------------------
    # Aggregate scores + threshold pass/fail per partition (AC-1, AC-3, AC-5)
    # ------------------------------------------------------------------
    aggregates: dict[str, dict[str, dict]] = {}
    all_pass = True

    for lang in sorted(results.keys()):
        lang_results = results[lang]
        lang_count = len(lang_results["faithfulness"])
        print(f"\n--- Aggregate ({lang.upper()} partition, N={lang_count}) ---")
        aggregates[lang] = {}
        for name, scores in lang_results.items():
            if not scores:
                aggregates[lang][name] = {"mean": None, "se": None, "threshold": METRIC_THRESHOLDS[name], "pass": False}
                print(f"  {name}: N/A")
                all_pass = False
                continue
            mean_val = float(np.mean(scores))
            se_val = float(np.std(scores) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
            threshold = METRIC_THRESHOLDS[name]
            passed = mean_val >= threshold
            if not passed:
                all_pass = False
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {name}: {mean_val:.4f} (±{se_val:.4f} SE)  (threshold: {threshold:.2f})  {status}")
            aggregates[lang][name] = {
                "mean": round(mean_val, 4),
                "se": round(se_val, 4),
                "threshold": threshold,
                "pass": passed,
            }

    overall = "✅ ALL PARTITIONS AND THRESHOLDS MET" if all_pass else "❌ ONE OR MORE THRESHOLDS NOT MET"
    print(f"\n  Overall: {overall}")
    print("\n[Disclaimer] Statistical Limitation (AC-5):")
    print("  At small sample sizes (e.g. N=17), standard error is approximately ±0.12, limiting the precision of per-partition conclusions.")

    # ------------------------------------------------------------------
    # Latency percentiles  (AC-4)
    # ------------------------------------------------------------------
    latency_stats: dict = {}
    if e2e_latencies:
        e2e_p = np.percentile(e2e_latencies, [50, 90, 95])
        ret_p = np.percentile(retrieval_latencies, [50, 90, 95])

        latency_stats = {
            "e2e_p50_s": round(float(e2e_p[0]), 3),
            "e2e_p90_s": round(float(e2e_p[1]), 3),
            "e2e_p95_s": round(float(e2e_p[2]), 3),
            "retrieval_p50_s": round(float(ret_p[0]), 3),
            "retrieval_p90_s": round(float(ret_p[1]), 3),
            "retrieval_p95_s": round(float(ret_p[2]), 3),
            "nfr_e2e_threshold_s": NFR_E2E_THRESHOLD_S,
            "nfr_retrieval_threshold_s": NFR_RETRIEVAL_THRESHOLD_S,
            "e2e_p95_pass": float(e2e_p[2]) <= NFR_E2E_THRESHOLD_S,
            "retrieval_p95_pass": float(ret_p[2]) <= NFR_RETRIEVAL_THRESHOLD_S,
        }

        e2e_status = "✅" if latency_stats["e2e_p95_pass"] else "❌"
        ret_status = "✅" if latency_stats["retrieval_p95_pass"] else "❌"

        print("\n--- Latency ---")
        print(f"  E2E       P50={e2e_p[0]:.2f}s  P90={e2e_p[1]:.2f}s  P95={e2e_p[2]:.2f}s  (≤{NFR_E2E_THRESHOLD_S}s) {e2e_status}")
        print(f"  Retrieval P50={ret_p[0]:.2f}s  P90={ret_p[1]:.2f}s  P95={ret_p[2]:.2f}s  (≤{NFR_RETRIEVAL_THRESHOLD_S}s) {ret_status}")
    # ------------------------------------------------------------------
    # Build structured JSON report
    # ------------------------------------------------------------------
    report: dict = {
        "timestamp": run_ts,
        "collection": collection,
        "evaluator": evaluator,
        "n_queries": len(EVAL_DATA),
        "n_scored": len(per_query) - sum(1 for q in per_query if q.get("skipped")),
        "thresholds": METRIC_THRESHOLDS,
        "aggregates": aggregates,
        "overall_pass": all_pass,
        "latency": latency_stats,
        "per_query": per_query,
    }
    # ------------------------------------------------------------------
    # Persist to evals/  (AC-1: JSON + summary → evals/)
    # ------------------------------------------------------------------
    evals_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "evals")
    os.makedirs(evals_dir, exist_ok=True)
    out_path = os.path.join(evals_dir, f"eval_{collection}_{run_ts}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n  Report saved → {out_path}")
    return report


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG Evaluation harness")
    parser.add_argument(
        "--evaluator", choices=["ollama", "gemini"], default="ollama",
        help="LLM to use as evaluator judge (default: ollama)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of benchmark queries to evaluate (default: None, evaluate all)",
    )
    args = parser.parse_args()

    run_evaluator(COLLECTION, evaluator=args.evaluator, limit=args.limit)