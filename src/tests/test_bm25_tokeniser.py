"""
test_bm25_tokeniser.py
======================
Compares the old naive tokeniser (text.lower().split()) against the new
Unicode-aware one (regex + punctuation stripping) in two ways:

  1. Token-level: shows how individual strings are tokenised differently.
  2. Retrieval-level: builds two BM25 indexes from a small Spanish corpus
     and shows whether relevant documents rank higher with the new tokeniser.

Run with:
    python3 -m src.tests.test_bm25_tokeniser
"""

import regex
from rank_bm25 import BM25Okapi


# ─── Tokenisers ──────────────────────────────────────────────────────────────

def old_tokenise(text: str) -> list[str]:
    """Old naive tokeniser: lowercase + whitespace split."""
    return text.lower().split()


def new_tokenise(text: str) -> list[str]:
    """
    New Unicode-aware tokeniser.
    Strips leading/trailing punctuation (\\p{P}) from each whitespace-split token.
    """
    tokens = []
    for tok in text.lower().split():
        tok = regex.sub(r'^\p{P}+|\p{P}+$', '', tok)
        if tok:
            tokens.append(tok)
    return tokens


# ─── Part 1: Token-level comparison ──────────────────────────────────────────

TOKEN_CASES = [
    # (input, description)
    ("¿Habitación?",         "Spanish open-question mark + trailing ?"),
    ("precio:",              "Trailing colon"),
    ('"glamping"',           "Double-quoted word"),
    ("naturaleza...",        "Trailing ellipsis"),
    ("¡Bienvenidos!",        "Spanish exclamation marks"),
    ("(cabañas)",            "Parentheses"),
    ("wifi,",               "Trailing comma"),
    ("check-in",             "Hyphenated compound (should stay intact)"),
    ("precio: 120€.",        "Full phrase with colon, currency and period"),
    ("¿Cuánto cuesta el glamping?", "Full question sentence"),
]


def run_token_comparison() -> None:
    COL_W = 40
    print("\n" + "=" * 80)
    print("PART 1 — Token-level comparison")
    print("=" * 80)
    print(f"{'Input':<{COL_W}}  {'Old tokens':<32}  New tokens")
    print("-" * 80)
    diffs = 0
    for text, desc in TOKEN_CASES:
        old = old_tokenise(text)
        new = new_tokenise(text)
        marker = " ✓" if old == new else " ←"
        if old != new:
            diffs += 1
        print(f"  {repr(text):<{COL_W-2}}  {str(old):<32}  {new}{marker}")
        print(f"  {desc}")
        print()
    print(f"Differences: {diffs}/{len(TOKEN_CASES)} cases changed")


# ─── Part 2: Retrieval-level comparison ──────────────────────────────────────

# Small corpus representing FAQ chunks (Spanish/Galician glamping context)
CORPUS = [
    "El precio del glamping incluye desayuno y acceso al spa.",
    "¿Cuánto cuesta la habitación para dos personas?",
    "Ofrecemos alojamiento en cabañas con vistas al río.",
    "El check-in es a partir de las 15:00 horas.",
    "Puede contactarnos por teléfono o correo electrónico.",
    "Las cabañas disponen de wifi gratuito y calefacción.",
    "El glamping está situado en un entorno natural privilegiado.",
    "Ofrecemos descuentos para estancias de más de tres noches.",
    "El restaurante sirve productos locales de la comarca.",
    "Los animales domésticos no están permitidos en las instalaciones.",
]

QUERIES = [
    ("¿Cuánto cuesta el glamping?",         "Price query (punctuation in question)"),
    ("habitación precio",                   "Vocabulary match with accented word"),
    ("check-in hora llegada",               "Hyphenated term"),
    ("wifi cabañas",                        "Keywords that appear mid-sentence"),
    ("¿Están permitidos los perros?",       "Negation + punctuation"),
]


def _bm25_top3(tokeniser, corpus: list[str], query: str) -> list[tuple[int, str, float]]:
    tokenised_corpus = [tokeniser(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(tokeniser(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
    return [(idx, corpus[idx], score) for idx, score in ranked]


def run_retrieval_comparison() -> None:
    print("\n" + "=" * 80)
    print("PART 2 — Retrieval-level comparison (top-3 per query)")
    print("=" * 80)

    wins_new = 0
    wins_old = 0

    for query, desc in QUERIES:
        old_results = _bm25_top3(old_tokenise, CORPUS, query)
        new_results = _bm25_top3(new_tokenise, CORPUS, query)

        print(f"\nQuery : {repr(query)}")
        print(f"Desc  : {desc}")
        print(f"  {'#':<4} {'OLD score':>10}  {'NEW score':>10}  Doc (truncated to 60 chars)")
        print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*60}")

        # Compare rank order
        old_ids = [r[0] for r in old_results]
        new_ids = [r[0] for r in new_results]

        for rank in range(3):
            oi, _, os = old_results[rank]
            ni, _, ns = new_results[rank]
            same = "=" if oi == ni else "≠"
            print(f"  #{rank+1}   old={os:8.4f}  new={ns:8.4f}  "
                  f"old→{oi} {same} new→{ni}  |  {CORPUS[ni][:60]}")

        if old_ids == new_ids:
            print("  → Same ranking")
        else:
            print("  → Rankings differ")
            # Heuristic: whichever tokeniser scores the #1 result higher wins
            if new_results[0][2] >= old_results[0][2]:
                wins_new += 1
                print("  → New tokeniser yields equal-or-higher top score ✓")
            else:
                wins_old += 1
                print("  → Old tokeniser yields higher top score")

    print(f"\nSummary: new tokeniser ≥ old in {wins_new}/{len(QUERIES)} queries "
          f"where rankings differed.")


# ─── Pytest cases ─────────────────────────────────────────────────────────────

def test_old_tokenise():
    assert old_tokenise("¿Habitación?") == ["¿habitación?"]
    assert old_tokenise("precio:") == ["precio:"]


def test_new_tokenise():
    assert new_tokenise("¿Habitación?") == ["habitación"]
    assert new_tokenise("precio:") == ["precio"]
    assert new_tokenise('"glamping"') == ["glamping"]
    assert new_tokenise("naturaleza...") == ["naturaleza"]
    assert new_tokenise("¡Bienvenidos!") == ["bienvenidos"]
    assert new_tokenise("(cabañas)") == ["cabañas"]
    assert new_tokenise("wifi,") == ["wifi"]
    assert new_tokenise("check-in") == ["check-in"]
    assert new_tokenise("precio: 120€.") == ["precio", "120€"]


def test_retrieval_comparison():
    # Make sure comparison function runs without errors
    run_retrieval_comparison()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_token_comparison()
    run_retrieval_comparison()
    print("\nDone.")
