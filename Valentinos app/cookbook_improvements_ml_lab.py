"""
Cookbook improvements & ML lab (standalone — does not import the Streamlit app).

Goals
-----
1. Regression-style tests for matching behaviour so the UI rarely ends up with
   zero recipes because of the coverage slider / strict filters.
2. Stronger, examinable ML than ingredient-only KNN: TF–IDF retrieval,
   optional logistic stacking, and topic discovery (NMF).

Course alignment (FCS-BWL Gruppenprojekt)
-----------------------------------------
- Requirement 5: the application must implement machine learning.
  This module demonstrates TF–IDF + cosine similarity (information retrieval),
  NMF (unsupervised decomposition), and a small supervised logistic model that
  learns a mapping from numeric features to a “good match” proxy label.

Integration
-----------
Copy the functions you want into ``cookbook_v2_1_test.py`` after validating
them here. Run this file directly: ``python cookbook_improvements_ml_lab.py``
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# Minimal offline “bonus” recipes for tests (subset of app idea)
# ─────────────────────────────────────────────

MINIMAL_BONUS: list[dict[str, Any]] = [
    {
        "id": "t_001",
        "name": "Tomato Pasta",
        "category": "Dinner",
        "area": "Italian",
        "ingredients": ["pasta", "tomatoes", "garlic", "olive oil", "salt"],
        "tags": "vegetarian",
    },
    {
        "id": "t_002",
        "name": "Egg Rice",
        "category": "Dinner",
        "area": "Asian",
        "ingredients": ["rice", "eggs", "soy sauce", "green onion"],
        "tags": "",
    },
    {
        "id": "t_003",
        "name": "Potato Soup",
        "category": "Starter",
        "area": "Swiss",
        "ingredients": ["potatoes", "onion", "butter", "milk", "salt"],
        "tags": "vegetarian",
    },
]


def compute_coverage(
    user_ings: Iterable[str], recipe_ings: Sequence[str]
) -> tuple[float, list[str], list[str]]:
    """Same idea as the app: fuzzy substring overlap per recipe ingredient."""
    recipe_set = {i.lower() for i in recipe_ings}
    user_set = {i.strip().lower() for i in user_ings if i and str(i).strip()}
    matched: set[str] = set()
    missing: set[str] = set()
    for r_ing in recipe_set:
        found = any(u in r_ing or r_ing in u for u in user_set)
        (matched if found else missing).add(r_ing)
    total = len(recipe_set)
    coverage = (len(matched) / total * 100) if total > 0 else 0.0
    return round(coverage, 1), sorted(matched), sorted(missing)


def _diet_ok(diet_filter: str, recipe: dict[str, Any]) -> bool:
    cat = str(recipe.get("category") or "").lower()
    tags = str(recipe.get("tags") or "").lower()
    ings = recipe.get("ingredients") or []
    ing_blob = " ".join(ings).lower()
    if diet_filter == "Vegetarian":
        if cat in ["beef", "chicken", "lamb", "pork", "seafood"]:
            return False
    if diet_filter == "Vegan":
        if cat in ["beef", "chicken", "lamb", "pork", "seafood", "dairy"]:
            return False
        if "vegan" not in tags and any(
            a in ing_blob
            for a in [
                "meat",
                "chicken",
                "beef",
                "pork",
                "fish",
                "egg",
                "milk",
                "cheese",
                "butter",
                "cream",
            ]
        ):
            return False
    return True


def _cuisine_ok(cuisine_filter: str, recipe: dict[str, Any]) -> bool:
    if cuisine_filter == "All":
        return True
    area = str(recipe.get("area") or "").lower()
    return cuisine_filter.lower() in area


def _source_rank_for_sort(recipe: dict[str, Any]) -> int:
    """Tie-break: defer hardcoded bonus when match scores tie (mirrors app)."""
    s = (recipe.get("source") or "api").lower()
    return {
        "spoonacular": 0,
        "api": 1,
        "forkify": 2,
        "hardcoded": 3,
    }.get(s, 1)


def match_recipes_no_coverage_filter(
    user_ingredients: list[str],
    all_recipes: list[dict[str, Any]],
    diet_filter: str = "All",
    cuisine_filter: str = "All",
) -> list[dict[str, Any]]:
    """
    Like the app’s matcher but **never** drops a recipe because of low coverage.
    Coverage is attached for display and optional sorting elsewhere.
    """
    user_set = {i.strip().lower() for i in user_ingredients if i.strip()}
    results: list[dict[str, Any]] = []
    for recipe in all_recipes:
        ings = recipe.get("ingredients") or []
        if not ings:
            continue
        if not _diet_ok(diet_filter, recipe):
            continue
        if not _cuisine_ok(cuisine_filter, recipe):
            continue
        coverage, matched, missing = compute_coverage(user_set, ings)
        row = {
            **recipe,
            "coverage": coverage,
            "matched": matched,
            "missing": missing,
            "missing_count": len(missing),
        }
        results.append(row)
    results.sort(
        key=lambda x: (
            -x["coverage"],
            x["missing_count"],
            _source_rank_for_sort(x),
        )
    )
    return results


def resolve_recipes_nonempty(
    user_ingredients: list[str],
    all_recipes: list[dict[str, Any]],
    diet_filter: str,
    cuisine_filter: str,
    min_coverage: float,
    strict_match_fn: Callable[..., list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str]:
    """
    Cascading fallback so the UI can **always** show something useful:

    1) Strict: same behaviour as the app (``strict_match_fn`` with min_coverage).
    2) Drop coverage gate: keep diet/cuisine, rank by coverage anyway.
    3) Widen filters: diet & cuisine ``All``, still no coverage gate.
    4) Last resort: return every recipe with ingredients (ignores filters),
       sorted by coverage — analytics can explain why matches are weak.

    ``strict_match_fn`` should have the signature
    ``(user_ingredients, all_recipes, diet, cuisine, min_coverage) -> list``.
    """
    if not user_ingredients:
        return [], "no_ingredients"

    strict = strict_match_fn(
        user_ingredients, all_recipes, diet_filter, cuisine_filter, min_coverage
    )
    if strict:
        return strict, "strict"

    relaxed = match_recipes_no_coverage_filter(
        user_ingredients, all_recipes, diet_filter, cuisine_filter
    )
    if relaxed:
        return relaxed, "no_coverage_gate"

    wide = match_recipes_no_coverage_filter(
        user_ingredients, all_recipes, "All", "All"
    )
    if wide:
        return wide, "widened_filters"

    bare = []
    for recipe in all_recipes:
        ings = recipe.get("ingredients") or []
        if not ings:
            continue
        cov, mat, mis = compute_coverage(
            {i.strip().lower() for i in user_ingredients if i.strip()}, ings
        )
        bare.append(
            {
                **recipe,
                "coverage": cov,
                "matched": mat,
                "missing": mis,
                "missing_count": len(mis),
            }
        )
    bare.sort(
        key=lambda x: (
            -x["coverage"],
            x["missing_count"],
            _source_rank_for_sort(x),
        )
    )
    return bare, "bare_minimum"


def _ingredient_document(recipe: dict[str, Any]) -> str:
    parts = list(recipe.get("ingredients") or [])
    parts += [str(recipe.get("name") or ""), str(recipe.get("tags") or "")]
    return " ".join(p.lower() for p in parts if p)


@dataclass
class IngredientRetrievalModel:
    """
    TF–IDF bag-of-words over ingredient lists + recipe metadata.
    Produces a dense relevance score in [0, 1] (cosine similarity clamped).
    """

    vectorizer: TfidfVectorizer
    recipe_matrix: Any

    @classmethod
    def fit(cls, recipes: list[dict[str, Any]]) -> IngredientRetrievalModel:
        docs = [_ingredient_document(r) for r in recipes]
        vec = TfidfVectorizer(
            max_features=4096,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w\w+\b",
        )
        matrix = vec.fit_transform(docs)
        return cls(vectorizer=vec, recipe_matrix=matrix)

    def scores(self, user_ingredients: list[str]) -> np.ndarray:
        q = self.vectorizer.transform([" ".join(user_ingredients).lower()])
        sim = cosine_similarity(q, self.recipe_matrix).ravel()
        # Cosine for TF–IDF rows is in [-1, 1]; clip for display.
        return np.clip(sim, 0.0, 1.0)


class MatchQualityClassifier:
    """
    Lightweight supervised model: learns which (coverage, missing ratio,
    TF–IDF similarity) tuples look like “good” matches.

    Labels are generated automatically: recipes with coverage >= label_threshold
    are positive. This is a **proxy** for user satisfaction used for ranking;
    replace with real ratings from your SQLite ``user_ratings`` table when
    you have enough data.
    """

    def __init__(self, label_threshold: float = 55.0):
        self.label_threshold = label_threshold
        self._pipe: Pipeline | None = None

    def fit(
        self,
        recipes: list[dict[str, Any]],
        user_ingredients: list[str],
        tfidf_scores: np.ndarray,
    ) -> MatchQualityClassifier:
        X_list = []
        y_list = []
        user_set = {i.strip().lower() for i in user_ingredients if i.strip()}
        for i, rec in enumerate(recipes):
            ings = rec.get("ingredients") or []
            if not ings:
                continue
            cov, _, miss = compute_coverage(user_set, ings)
            miss_ratio = len(miss) / max(len(ings), 1)
            feat = [cov / 100.0, 1.0 - miss_ratio, float(tfidf_scores[i])]
            X_list.append(feat)
            y_list.append(1 if cov >= self.label_threshold else 0)
        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=int)
        if len(np.unique(y)) < 2 or len(y) < 6:
            # Not enough signal: fall back to identity coefficients.
            self._pipe = None
            return self
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self._pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=200, class_weight="balanced"),
                ),
            ]
        )
        self._pipe.fit(X_train, y_train)
        return self

    def proba_good(self, features_rows: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            # Heuristic fallback: weighted sum of normalized features.
            w = np.array([0.35, 0.35, 0.30])
            s = (features_rows * w).sum(axis=1)
            return 1.0 / (1.0 + np.exp(-5 * (s - 0.5)))

        return self._pipe.predict_proba(features_rows)[:, 1]


def nmf_ingredient_topics(
    recipes: list[dict[str, Any]], n_topics: int = 3, random_state: int = 0
) -> tuple[Any, TfidfVectorizer, list[list[tuple[str, float]]]]:
    """
    Unsupervised NMF over a TF–IDF document-term matrix of ingredients.
    Returns (W, vectorizer, top_terms_per_topic) for explanations in the UI.
    """
    docs = [_ingredient_document(r) for r in recipes]
    vec = TfidfVectorizer(
        max_features=256,
        min_df=1,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    X = vec.fit_transform(docs)
    n_topics = min(n_topics, X.shape[1], X.shape[0])
    if n_topics < 2:
        return None, vec, []
    model = NMF(n_components=n_topics, init="nndsvda", random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    terms = np.array(vec.get_feature_names_out())
    topics: list[list[tuple[str, float]]] = []
    for k in range(n_topics):
        top_idx = np.argsort(H[k])[-8:][::-1]
        topics.append([(terms[j], float(H[k, j])) for j in top_idx])
    return W, vec, topics


def rank_recipes_ml_blend(
    user_ingredients: list[str],
    recipes: list[dict[str, Any]],
    retrieval: IngredientRetrievalModel,
    quality: MatchQualityClassifier | None = None,
    weight_tfidf: float = 0.45,
    weight_quality: float = 0.35,
    weight_coverage: float = 0.20,
) -> list[dict[str, Any]]:
    """
    Final ordering: blend TF–IDF similarity, optional learned “quality” proba,
    and coverage. No recipe is removed; low coverage simply sinks in the list.
    """
    if not recipes:
        return []
    tfidf_scores = retrieval.scores(user_ingredients)
    if quality is None:
        quality = MatchQualityClassifier().fit(recipes, user_ingredients, tfidf_scores)

    user_set = {i.strip().lower() for i in user_ingredients if i.strip()}
    rows: list[dict[str, Any]] = []
    feat_rows = []
    for i, rec in enumerate(recipes):
        ings = rec.get("ingredients") or []
        cov, matched, miss = compute_coverage(user_set, ings)
        miss_ratio = len(miss) / max(len(ings), 1)
        fr = np.array([[cov / 100.0, 1.0 - miss_ratio, float(tfidf_scores[i])]])
        feat_rows.append(fr[0])
        rows.append(
            {
                **rec,
                "coverage": cov,
                "matched": matched,
                "missing": miss,
                "missing_count": len(miss),
                "ml_tfidf": round(float(tfidf_scores[i]), 4),
            }
        )
    F = np.vstack(feat_rows)
    q_good = quality.proba_good(F)
    cov_norm = F[:, 0]
    for i, r in enumerate(rows):
        blend = (
            weight_tfidf * float(tfidf_scores[i])
            + weight_quality * float(q_good[i])
            + weight_coverage * float(cov_norm[i])
        )
        r["ml_rank_score"] = round(float(blend), 4)
        r["ml_quality_proba"] = round(float(q_good[i]), 4)
    rows.sort(key=lambda x: -x["ml_rank_score"])
    return rows


# ─────────────────────────────────────────────
# Strict matcher compatible with ``resolve_recipes_nonempty`` (mirrors app)
# ─────────────────────────────────────────────


def match_recipes_strict_gate(
    user_ingredients: list[str],
    all_recipes: list[dict[str, Any]],
    diet_filter: str = "All",
    cuisine_filter: str = "All",
    min_coverage: float = 20.0,
) -> list[dict[str, Any]]:
    """Same filtering behaviour as ``match_recipes`` in the Streamlit app."""
    user_set = {i.strip().lower() for i in user_ingredients if i.strip()}
    results: list[dict[str, Any]] = []
    for recipe in all_recipes:
        ings = recipe.get("ingredients") or []
        if not ings:
            continue
        coverage, matched, missing = compute_coverage(user_set, ings)
        if coverage < min_coverage:
            continue
        if not _diet_ok(diet_filter, recipe):
            continue
        if not _cuisine_ok(cuisine_filter, recipe):
            continue
        results.append(
            {
                **recipe,
                "coverage": coverage,
                "matched": matched,
                "missing": missing,
                "missing_count": len(missing),
            }
        )
    results.sort(
        key=lambda x: (
            -x["coverage"],
            x["missing_count"],
            _source_rank_for_sort(x),
        )
    )
    return results


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────


class TestCoverageFallback(unittest.TestCase):
    def test_strict_empty_then_fallback_finds_recipes(self):
        """Exotic pantry + high min_coverage → strict empty; fallback still lists."""
        user = ["dragonfruit", "quinoa", "truffle oil"]
        strict = match_recipes_strict_gate(
            user, MINIMAL_BONUS, "All", "All", min_coverage=80
        )
        self.assertEqual(len(strict), 0)
        resolved, mode = resolve_recipes_nonempty(
            user,
            MINIMAL_BONUS,
            "Vegan",
            "Italian",
            min_coverage=80,
            strict_match_fn=match_recipes_strict_gate,
        )
        self.assertGreater(len(resolved), 0, mode)
        self.assertIn(mode, ("no_coverage_gate", "widened_filters", "bare_minimum"))

    def test_no_coverage_filter_keeps_recipes_with_low_overlap(self):
        user = ["salt"]
        out = match_recipes_no_coverage_filter(user, MINIMAL_BONUS, "All", "All")
        self.assertGreater(len(out), 0, "Should still list recipes; coverage is informational only.")

    def test_ml_ranking_is_deterministic(self):
        user = ["pasta", "tomatoes", "garlic"]
        retrieval = IngredientRetrievalModel.fit(MINIMAL_BONUS)
        ranked = rank_recipes_ml_blend(user, MINIMAL_BONUS, retrieval)
        names = [r["name"] for r in ranked]
        again = rank_recipes_ml_blend(user, MINIMAL_BONUS, retrieval)
        self.assertEqual(names, [r["name"] for r in again])


class TestNMF(unittest.TestCase):
    def test_nmf_runs_on_toy_corpus(self):
        W, vec, topics = nmf_ingredient_topics(MINIMAL_BONUS, n_topics=2)
        self.assertIsNotNone(vec)
        if topics:
            self.assertTrue(all(isinstance(t, tuple) for row in topics for t in row))


def _demo_console_report() -> None:
    user = ["rice", "egg"]
    print("=== Demo: fallback chain ===")
    resolved, mode = resolve_recipes_nonempty(
        user,
        MINIMAL_BONUS,
        diet_filter="Vegan",
        cuisine_filter="Italian",
        min_coverage=95,
        strict_match_fn=match_recipes_strict_gate,
    )
    print("mode:", mode, "count:", len(resolved))
    for r in resolved[:5]:
        print(
            f"  - {r['name']}: coverage={r['coverage']}% "
            f"missing={r['missing_count']}"
        )

    print("\n=== Demo: ML blend ranking ===")
    retrieval = IngredientRetrievalModel.fit(MINIMAL_BONUS)
    ranked = rank_recipes_ml_blend(user, MINIMAL_BONUS, retrieval)
    for r in ranked:
        print(
            f"  - {r['name']}: ml_rank={r['ml_rank_score']} "
            f"tfidf={r['ml_tfidf']} coverage={r['coverage']}%"
        )

    print("\n=== Demo: NMF topics (ingredient co-occurrence structure) ===")
    _, _, topics = nmf_ingredient_topics(MINIMAL_BONUS, n_topics=2)
    for i, tlist in enumerate(topics):
        terms = ", ".join(f"{w}({s:.2f})" for w, s in tlist[:5])
        print(f"  Topic {i+1}: {terms}")


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
    _demo_console_report()
