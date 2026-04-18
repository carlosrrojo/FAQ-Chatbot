"""
RAG Evaluation harness.

Evaluates the retrieve → generate pipeline using four metrics:
  - Faithfulness      (answer supported by retrieved context)
  - Answer Relevancy  (answer addresses the question)
  - Context Precision (retrieved chunks are useful)
  - Context Recall    (context covers the ground-truth answer)
"""

from sqlalchemy.orm.collections import collection
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from benchmarks.eval_data import DATA
from langchain_core.globals import set_debug
from src.rag.agent import generate_reply, retrieve_documents
#from src.rag.rag_as_agent import generate_reply, retrieve_documents

set_debug(False)

EVAL_DATA = DATA


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _parse_score(text: str) -> float:
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
    try:
        if "```json" in res:
            res = res.split("```json")[-1].split("```")[0]
        gen_questions = json.loads(res)
    except Exception:
        gen_questions = [q.strip("- 1234567890.") for q in res.split("\n") if q.strip()]

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

def run_evaluator(collection: str = "Espazo Nature") -> None:
    print(f"\n{'='*44}")
    print(f"Collection: {collection}")
    print(f"{'='*44}")

    model      = ChatOllama(model="llama3.1")
    embeddings = OllamaEmbeddings(model="llama3.1")

    results: dict[str, list[float]] = {
        "faithfulness":    [],
        "answer_relevancy":[],
        "context_precision":[],
        "context_recall":  [],
    }

    for idx, item in enumerate(EVAL_DATA):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        # .func calls the underlying Python function directly, bypassing the
        # tool wrapper — returns the raw (serialized_str, docs_list) 2-tuple
        _, retrieved_docs = retrieve_documents.func(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        answer = generate_reply("whatsapp", question, "eval-runner")
        
        if not ground_truth:
            print("  ⚠ No ground truth — skipping metrics.")
            continue

        f_score  = calculate_faithfulness(question, answer, contexts, model)
        ar_score = calculate_answer_relevancy(question, answer, model, embeddings)
        cp_score = calculate_context_precision(question, ground_truth, retrieved_docs, model)
        cr_score = calculate_context_recall(question, ground_truth, contexts, model)

        print(f"\nQ{idx+1}/{len(EVAL_DATA)}: {question} |F: {f_score:.2f} | AR: {ar_score:.2f} | CP: {cp_score:.2f} | CR: {cr_score:.2f}")
        #print(f"\nQ{idx+1}/{len(EVAL_DATA)}: {question}")
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
        results["faithfulness"].append(f_score)
        results["answer_relevancy"].append(ar_score)
        results["context_precision"].append(cp_score)
        results["context_recall"].append(cr_score)

    print(f"\n--- Aggregate ({collection}) ---")
    for name, scores in results.items():
        print(f"  {name}: {np.mean(scores):.4f}" if scores else f"  {name}: N/A")


if __name__ == "__main__":
    load_dotenv()
    sizes = [(512, 64),(800,100),(1024, 256)]
    """for c_s, c_o in sizes:
        collection = f"metadata_espazo_nature_{c_s}_{c_o}"""
    collection = "metadata_NoHybrid_NoRerank_1024"
    run_evaluator(collection)