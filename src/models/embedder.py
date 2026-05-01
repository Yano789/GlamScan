"""
src/models/embedder.py

Zero-shot CLIP embedder.
Wraps open_clip to produce L2-normalised 512-d image embeddings.

No fine-tuning required — CLIP's visual encoder already generalises
extremely well to cosmetic product images out of the box.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
import open_clip
from PIL import Image

from src.utils.config import CLIP_MODEL, CLIP_PRETRAINED, EMBEDDING_DIM
from src.utils.logger import get_logger

log = get_logger("embedder")


class CLIPEmbedder:
    """
    Singleton-friendly CLIP wrapper.

    Usage:
        embedder = CLIPEmbedder()
        vec = embedder.embed_image(pil_img)         # np.ndarray (512,)
        vecs = embedder.embed_images([img1, img2])  # np.ndarray (N, 512)
        text_vec = embedder.embed_text("red lipstick")
    """

    def __init__(
        self,
        model_name:  str = CLIP_MODEL,
        pretrained:  str = CLIP_PRETRAINED,
        device:      str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading CLIP %s/%s on %s …", model_name, pretrained, self.device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(self.device)

        log.info("CLIP ready  (embedding dim: %d)", EMBEDDING_DIM)

    # ── Image embedding ────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single PIL image → L2-normalised float32 vector."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feat   = self.model.encode_image(tensor)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().float().numpy()[0]

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed a list of PIL images in batches → (N, D) float32 array."""
        all_vecs = []
        for i in range(0, len(images), batch_size):
            batch  = images[i : i + batch_size]
            tensor = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            feat   = self.model.encode_image(tensor)
            feat   = feat / feat.norm(dim=-1, keepdim=True)
            all_vecs.append(feat.cpu().float().numpy())
        return np.vstack(all_vecs)

    # ── Text embedding (for zero-shot queries) ─────────────────────────────────
    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query → L2-normalised float32 vector (same space as images)."""
        tokens = self.tokenizer([text]).to(self.device)
        feat   = self.model.encode_text(tokens)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().float().numpy()[0]

    # ── Convenience: embed from path or bytes ─────────────────────────────────
    def embed_path(self, path: Union[str, Path]) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return self.embed_image(img)

    def embed_bytes(self, data: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return self.embed_image(img)


# ── Module-level singleton (lazy) ─────────────────────────────────────────────
_embedder: CLIPEmbedder | None = None


def get_embedder() -> CLIPEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = CLIPEmbedder()
    return _embedder
