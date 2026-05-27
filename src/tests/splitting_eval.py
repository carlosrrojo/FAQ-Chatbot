import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

import src.utils as utils
from src.config import DB_PATH, COLLECTION, EMBEDDING_MODEL
from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# -------------------------------
# LLM RELEVANCE GRADER
# -------------------------------

def grade_relevance(query: str, chunk: str, llm) -> int:
    """
    Returns relevance score from 0 to 3 using LLM grading.
    """
    prompt = f"""
You are evaluating retrieval quality for a RAG system.

Query:
{query}

Chunk:
{chunk}

Score relevance from 0 to 3:

0 = Not relevant at all
1 = Slightly relevant
2 = Mostly relevant
3 = Highly relevant and directly answers the query

Return ONLY the number.
"""
    try:
        response_msg = llm.invoke(prompt)
        response_text = response_msg.content.strip()
        match = re.search(r"[0-3]", response_text)
        if match:
            return int(match.group())
        logger.warning("Could not extract a valid score (0-3) from response: %r", response_text)
        return 0
    except Exception:
        logger.exception("Failed to grade relevance for query %r", query)
        return 0


# -------------------------------
# METRIC COMPUTATION
# -------------------------------

def precision_at_k(scores: List[int], k: int, threshold: int) -> float:
    if not scores:
        return 0.0
    relevant = sum(1 for s in scores[:k] if s >= threshold)
    return relevant / min(len(scores), k)


def reciprocal_rank(scores: List[int], threshold: int) -> float:
    for i, s in enumerate(scores):
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


# -------------------------------
# MAIN EVALUATION FUNCTION
# -------------------------------

def run_splitting_eval(
    collection_name: str,
    db_path: str,
    evaluator: str,
    benchmark_file: str,
    top_k: int,
    relevance_threshold: int
) -> dict:
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("=== RAG Chunking Evaluation ===")
    logger.info("Collection: %s", collection_name)
    logger.info("Database Path: %s", db_path)
    logger.info("Evaluator LLM: %s", evaluator)
    logger.info("Benchmark: %s", benchmark_file)
    logger.info("Top K: %d | Relevance Threshold: %d", top_k, relevance_threshold)

    # 1. Setup Models
    if evaluator == "gemini":
        model = ChatGoogleGenerativeAI(model="gemini-flash-latest")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        model = ChatOllama(model="llama3.1")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 2. Setup Vector Store
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    # Check collection size
    try:
        col_count = vectorstore._collection.count()
        logger.info("Collection document count: %d", col_count)
        if col_count == 0:
            logger.warning("Chroma collection '%s' is empty!", collection_name)
    except Exception:
        logger.exception("Failed to query collection count.")

    # 3. Load benchmark queries
    test_queries = utils.load_benchmark(benchmark_file)
    logger.info("Loaded %d queries.", len(test_queries))

    all_precisions = []
    all_mrr = []
    all_mean_scores = []
    per_query_results = []

    for idx, query in enumerate(test_queries):
        logger.info("\nQuery %d/%d: %s", idx + 1, len(test_queries), query)
        
        # Retrieve chunks
        docs: List[Document] = vectorstore.similarity_search(query, k=top_k)
        
        if not docs:
            logger.warning("No chunks retrieved for query.")
            p_at_k = 0.0
            rr = 0.0
            mean_score = 0.0
            scores = []
        else:
            scores = []
            for i, doc in enumerate(docs):
                score = grade_relevance(query, doc.page_content, model)
                scores.append(score)
                logger.info("  - Chunk %d | Score: %d | Content: %s...", i + 1, score, doc.page_content[:80].replace("\n", " "))
            
            p_at_k = precision_at_k(scores, top_k, relevance_threshold)
            rr = reciprocal_rank(scores, relevance_threshold)
            mean_score = float(np.mean(scores))

        all_precisions.append(p_at_k)
        all_mrr.append(rr)
        all_mean_scores.append(mean_score)

        logger.info("  Precision@%d: %.2f", top_k, p_at_k)
        logger.info("  Reciprocal Rank: %.2f", rr)
        logger.info("  Mean Relevance: %.2f", mean_score)

        per_query_results.append({
            "query": query,
            "scores": scores,
            "precision_at_k": round(p_at_k, 4),
            "reciprocal_rank": round(rr, 4),
            "mean_relevance": round(mean_score, 4) if not np.isnan(mean_score) else None
        })

        # Throttle Gemini API calls to avoid rate limits
        if evaluator == "gemini" and idx < len(test_queries) - 1:
            time.sleep(5)

    # 4. Final Aggregates
    mean_precision = float(np.mean(all_precisions)) if all_precisions else 0.0
    mean_mrr = float(np.mean(all_mrr)) if all_mrr else 0.0
    mean_relevance = float(np.mean([s for s in all_mean_scores if not np.isnan(s)])) if all_mean_scores else 0.0

    logger.info("\n=== OVERALL RESULTS ===")
    logger.info("Mean Precision@%d: %.3f", top_k, mean_precision)
    logger.info("Mean MRR: %.3f", mean_mrr)
    logger.info("Mean Relevance Score: %.3f", mean_relevance)

    report = {
        "timestamp": run_ts,
        "collection": collection_name,
        "evaluator": evaluator,
        "benchmark": benchmark_file,
        "top_k": top_k,
        "relevance_threshold": relevance_threshold,
        "mean_precision": round(mean_precision, 4),
        "mean_mrr": round(mean_mrr, 4),
        "mean_relevance": round(mean_relevance, 4),
        "per_query": per_query_results
    }

    # 5. Persist report to evals/
    evals_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "evals")
    os.makedirs(evals_dir, exist_ok=True)
    out_path = os.path.join(evals_dir, f"chunk_eval_{collection_name}_{run_ts}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Evaluation report saved to: %s", out_path)

    return report


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate quality of document chunking strategy")
    parser.add_argument(
        "--collection", default=COLLECTION,
        help=f"Chroma collection name (default: {COLLECTION})"
    )
    parser.add_argument(
        "--db-path", default=DB_PATH,
        help=f"Path to Chroma DB directory (default: {DB_PATH})"
    )
    parser.add_argument(
        "--evaluator", choices=["ollama", "gemini"], default="ollama",
        help="LLM to use as evaluator judge (default: ollama)"
    )
    parser.add_argument(
        "--benchmark", default="alojamientos.txt",
        help="Benchmark query file in benchmarks/ directory (default: alojamientos.txt)"
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of top chunks to retrieve (default: 3)"
    )
    parser.add_argument(
        "--relevance-threshold", type=int, default=2,
        help="Minimum LLM-relevance score (0-3) to consider a chunk relevant (default: 2)"
    )

    args = parser.parse_args()

    run_splitting_eval(
        collection_name=args.collection,
        db_path=args.db_path,
        evaluator=args.evaluator,
        benchmark_file=args.benchmark,
        top_k=args.k,
        relevance_threshold=args.relevance_threshold
    )
