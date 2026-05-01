"""
src/retrieval/search.py

Similarity search over the FAISS index.

Supports:
  - Image query  (PIL Image or bytes)
  - Text query   (zero-shot, e.g. "matte red lipstick")
  - Combined     (text + image, averaged embeddings)
  - Color-aware filtering for visual relevance
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import faiss
import numpy as np
from PIL import Image

from src.models.embedder import get_embedder
from src.utils.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K, MIN_SIMILARITY
from src.utils.logger import get_logger

log = get_logger("search")


@dataclass
class SearchResult:
    rank:        int
    score:       float          # cosine similarity (0–1)
    product_id:  str
    name:        str
    brand:       str
    category:    str
    price:       str
    price_usd:   float | None
    source:      str
    url:         str
    image_url:   str
    rating:      str
    reviews:     str


# ── Color extraction helpers ───────────────────────────────────────────────────
def _extract_dominant_color(image: Image.Image, n_colors: int = 3) -> np.ndarray:
    """Extract dominant color from image as RGB vector (0-1 scale)."""
    try:
        # Resize for speed
        img_small = image.resize((50, 50))
        img_array = np.array(img_small).astype(np.float32) / 255.0
        
        # Reshape to 2D: (50*50, 3)
        pixels = img_array.reshape(-1, 3)
        
        # Simple: return mean color (dominant across image)
        mean_color = pixels.mean(axis=0)
        return mean_color
    except Exception as e:
        log.debug("Color extraction failed: %s. Using neutral gray.", e)
        return np.array([0.5, 0.5, 0.5])


def _color_distance(color1: np.ndarray, color2: np.ndarray) -> float:
    """Euclidean distance in RGB space (0 = same, sqrt(3) = max)."""
    diff = np.array(color1) - np.array(color2)
    return np.linalg.norm(diff)


def _color_name(rgb: np.ndarray) -> str:
    """Convert RGB (0-1 scale) to approximate color name."""
    r, g, b = rgb
    
    # Brightness
    brightness = (r + g + b) / 3
    
    # If very light or very dark
    if brightness > 0.9:
        return "white"
    if brightness < 0.1:
        return "black"
    
    # Find dominant channel
    max_idx = np.argmax([r, g, b])
    
    if max_idx == 0:  # Red dominant
        if g > 0.5 and b < 0.3:
            return "orange"
        if g > 0.4 and b > 0.4:
            return "pink"
        return "red"
    elif max_idx == 1:  # Green dominant
        if r > 0.3 and b < 0.3:
            return "brown"
        if r < 0.3 and b < 0.3:
            return "green"
        return "yellow"
    else:  # Blue dominant
        if r > 0.3 and g > 0.3:
            return "purple"
        if r < 0.3 and g < 0.3:
            return "blue"
        return "purple"


def _apply_color_filter(
    results: list[SearchResult],
    query_color: np.ndarray,
    metadata: list[dict],
    color_weight: float = 0.15,
) -> list[SearchResult]:
    """Re-weight results based on color similarity to query image."""
    for result in results:
        try:
            # Get product metadata with image
            if result.image_url:
                # For now, we can't fetch external images in search phase
                # This is a placeholder for future enhancement
                pass
        except Exception:
            pass
    
    # For now, color filtering is a placeholder
    # In production, you'd cache product colors during indexing
    return results


class GlamSearchEngine:
    """
    Loads the FAISS index and metadata once, then serves queries.
    Thread-safe for reading (FAISS read-only searches are thread-safe).
    """

    def __init__(self) -> None:
        self._index    : faiss.Index | None = None
        self._metadata : list[dict]          = []
        self._n_products = 0
        self._embedding_dim = 512

    def _load(self) -> None:
        if self._index is not None:
            return  # already loaded

        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run `python -m src.retrieval.build_index` first."
            )
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {METADATA_PATH}. "
                "Run `python -m src.models.infer_embedder` first."
            )

        self._index    = faiss.read_index(str(FAISS_INDEX_PATH))
        self._metadata = json.loads(METADATA_PATH.read_text())
        self._n_products = len(self._metadata)
        self._embedding_dim = self._index.d if hasattr(self._index, 'd') else 512
        log.info("Search engine ready: %d products", self._n_products)

    @property
    def n_products(self) -> int:
        """Number of products in the index."""
        if self._index is None:
            self._load()
        return self._n_products
    
    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        if self._index is None:
            self._load()
        return self._embedding_dim

    # ── Query helpers ──────────────────────────────────────────────────────────
    def _query(self, vec: np.ndarray, top_k: int) -> list[SearchResult]:
        self._load()
        vec = vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)

        # Fetch more results to allow for category filtering
        k_fetch = min(top_k * 3, 100)
        scores, indices = self._index.search(vec, k_fetch)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._metadata):
                continue
            m = self._metadata[idx]
            results.append(SearchResult(
                rank       = rank,
                score      = float(score),
                product_id = m.get("product_id", ""),
                name       = m.get("name", ""),
                brand      = m.get("brand", ""),
                category   = m.get("category", ""),
                price      = m.get("price", ""),
                price_usd  = m.get("price_usd"),
                source     = m.get("source", ""),
                url        = m.get("url", ""),
                image_url  = m.get("image_url", ""),
                rating     = m.get("rating", ""),
                reviews    = m.get("reviews", ""),
            ))
        return results[:top_k]  # Return only top_k

    # ── Public API ─────────────────────────────────────────────────────────────
    def search_by_image(
        self,
        image: Union[Image.Image, bytes],
        top_k: int = TOP_K,
    ) -> list[SearchResult]:
        t0 = time.time()
        embedder = get_embedder()
        log.info(f"Embedder loaded: {time.time() - t0:.2f}s")
        
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Extract query color
        query_color = _extract_dominant_color(image)
        query_color_name = _color_name(query_color)
        
        t1 = time.time()
        vec = embedder.embed_image(image)
        log.info(f"Image embedding: {time.time() - t1:.2f}s")
        
        # *** NEW: Detect product type from image using CLIP zero-shot classification ***
        t_classify = time.time()
        detected_category = self._classify_product_type_from_image(embedder, image)
        log.info(f"Product type detection: {time.time() - t_classify:.2f}s → {detected_category}")
        
        t2 = time.time()
        results = self._query(vec, top_k * 3)  # Fetch more to filter
        
        # Apply product type filtering + aggressive category penalty
        filtered_results = []
        for r in results:
            name_lower = r.name.lower()
            
            # STRONG: If we detected a category, heavily penalize mismatches
            if detected_category and r.category.lower() != detected_category:
                r.score *= 0.2  # Even heavier penalty (80% reduction) when type detected
            
            # Penalize conflicting colors
            conflicting_colors = {
                "red": ["white", "clear", "nude", "transparent"],
                "pink": ["white", "clear", "nude", "black", "gray"],
                "brown": ["white", "clear", "nude", "pink"],
                "nude": ["black", "red", "pink", "brown", "purple"],
            }
            
            if query_color_name in conflicting_colors:
                for conflict_color in conflicting_colors[query_color_name]:
                    if conflict_color in name_lower:
                        r.score *= 0.7
                        break
            
            filtered_results.append(r)
        
        # Re-sort by score
        filtered_results = sorted(filtered_results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(filtered_results[:top_k], 1):
            r.rank = i
        
        # Apply similarity threshold
        filtered_results = [r for r in filtered_results if r.score >= MIN_SIMILARITY]
        
        log.info(f"FAISS query + category+color filter: {time.time() - t2:.2f}s ({len(filtered_results)} results)")
        
        return filtered_results[:top_k]

    def search_by_text(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[SearchResult]:
        """Search by text query with category-aware re-ranking."""
        embedder = get_embedder()
        vec = embedder.embed_text(query)
        
        # Get raw results (fetch more for filtering)
        results = self._query(vec, top_k * 2)
        
        # Extract category from query and boost matching results
        inferred_cat = self._infer_category(query)
        if inferred_cat:
            for result in results:
                if result.category.lower() == inferred_cat.lower():
                    result.score *= 1.2  # Boost matching category by 20% for text search
        
        # Re-sort and return top_k
        results = sorted(results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results[:top_k], 1):
            r.rank = i
        
        return results[:top_k]

    def search_by_combined(
        self,
        image: Union[Image.Image, bytes],
        text:  str,
        top_k: int = TOP_K,
        text_weight: float = 0.5,
    ) -> list[SearchResult]:
        """Blend image and text embeddings with category and color awareness."""
        embedder = get_embedder()
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Extract query color and include in text for semantic understanding
        query_color = _extract_dominant_color(image)
        color_name = _color_name(query_color)
        
        # Augment text query with detected color if not already specified
        if color_name not in text.lower():
            text_with_color = f"{text} {color_name}"
        else:
            text_with_color = text
        
        img_vec  = embedder.embed_image(image)
        text_vec = embedder.embed_text(text_with_color)
        vec      = (1 - text_weight) * img_vec + text_weight * text_vec
        
        # Get raw results (we fetch more to allow filtering)
        results = self._query(vec, top_k * 2)
        
        # Extract category from text query
        inferred_cat = self._infer_category(text)
        
        # Re-rank: boost score if category matches
        if inferred_cat:
            for result in results:
                if result.category.lower() == inferred_cat.lower():
                    result.score *= 1.1
        
        # Apply color-based filtering
        for r in results:
            name_lower = r.name.lower()
            conflicting_colors = {
                "red": ["white", "clear", "nude", "transparent"],
                "pink": ["white", "clear", "nude", "black", "gray"],
                "brown": ["white", "clear", "nude", "pink"],
                "nude": ["black", "red", "pink", "brown", "purple"],
            }
            
            if color_name in conflicting_colors:
                for conflict_color in conflicting_colors[color_name]:
                    if conflict_color in name_lower:
                        r.score *= 0.7
                        break
        
        # Re-sort and return top_k
        results = sorted(results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results[:top_k], 1):
            r.rank = i
        
        # Apply similarity threshold since image is used
        results = [r for r in results if r.score >= MIN_SIMILARITY]
        
        return results[:top_k]

    @staticmethod
    def _infer_category(query: str) -> str | None:
        """Extract product category from query text."""
        query_lower = query.lower()
        categories = ["blush", "lipstick", "mascara", "foundation", "eyeshadow", "concealer", "primer", "setting_spray"]
        
        for cat in categories:
            if cat in query_lower or cat.replace("_", " ") in query_lower:
                return cat
        return None
    
    def _classify_product_type_from_image(self, embedder, image: Image.Image) -> str | None:
        """
        Use CLIP zero-shot classification to detect product type from image.
        
        Returns the most likely product category (lipstick, blush, foundation, etc.)
        by comparing image embedding to category text embeddings.
        """
        try:
            # Get image embedding
            img_embedding = embedder.embed_image(image)
            
            # Define category templates
            categories = ["blush", "lipstick", "mascara", "foundation", "eyeshadow", "concealer", "primer", "setting_spray"]
            category_prompts = {
                "lipstick": "a lipstick product",
                "mascara": "a mascara product",
                "foundation": "a foundation product",
                "eyeshadow": "an eyeshadow product",
                "blush": "a blush product",
                "concealer": "a concealer product",
                "primer": "a primer product",
                "setting_spray": "a setting spray product",
            }
            
            # Get embeddings for each category
            category_embeddings = {}
            for cat, prompt in category_prompts.items():
                cat_embedding = embedder.embed_text(prompt)
                category_embeddings[cat] = cat_embedding
            
            # Compute similarity between image and each category
            similarities = {}
            for cat, cat_embedding in category_embeddings.items():
                # Normalize and compute cosine similarity
                img_norm = img_embedding / (np.linalg.norm(img_embedding) + 1e-8)
                cat_norm = cat_embedding / (np.linalg.norm(cat_embedding) + 1e-8)
                similarity = float(np.dot(img_norm, cat_norm))
                similarities[cat] = similarity
            
            # Get the best match
            best_category = max(similarities, key=similarities.get)
            best_score = similarities[best_category]
            
            # Only return if confidence is high (>0.25, typical for CLIP similarities)
            if best_score > 0.20:
                log.debug(f"Product type classification: {best_category} (score: {best_score:.3f})")
                log.debug(f"All scores: {similarities}")
                return best_category
            else:
                log.debug(f"Product type classification confidence too low: {best_score:.3f}")
                return None
        except Exception as e:
            log.warning(f"Product type classification failed: {e}")
            return None
    
    @staticmethod
    def _infer_category_from_results(results: list[SearchResult], query_color: str) -> str | None:
        """
        Infer product category from top 3-5 results using consensus + color heuristics.
        
        Returns the most common category among top results.
        If tied or unclear, uses color priors (e.g., red → lipstick, brown → foundation).
        """
        if not results:
            return None
        
        # Get categories from top 3-5 results
        top_n = min(5, len(results))
        top_categories = [r.category.lower() for r in results[:top_n]]
        
        # Count occurrences
        from collections import Counter
        category_counts = Counter(top_categories)
        
        # If one category dominates (appears in 3+ of top 5), use it
        most_common_cat, count = category_counts.most_common(1)[0]
        if count >= 3:
            return most_common_cat
        
        # Otherwise, use color heuristics to break ties
        color_hints = {
            "red":    "lipstick",      # Red typically means lipstick
            "pink":   "blush",         # Pink typically means blush
            "brown":  "foundation",    # Brown typically means foundation
            "nude":   "foundation",    # Nude typically means foundation
            "black":  "mascara",       # Black typically means mascara
            "purple": "eyeshadow",     # Purple can mean eyeshadow or lipstick, but eyeshadow often has purple
            "green":  "eyeshadow",     # Green is typically eyeshadow
            "blue":   "eyeshadow",     # Blue is typically eyeshadow
            "white":  "primer",        # White often means primer/base
        }
        
        if query_color in color_hints:
            hinted_cat = color_hints[query_color]
            # Use hint only if it appears in top results
            if hinted_cat in top_categories:
                return hinted_cat
            # If hint doesn't appear in top results, fall back to most common
            return most_common_cat
        
        return most_common_cat


# ── Module-level singleton ─────────────────────────────────────────────────────
_engine: GlamSearchEngine | None = None


def get_engine() -> GlamSearchEngine:
    global _engine
    if _engine is None:
        _engine = GlamSearchEngine()
    return _engine
