"""
app/services/qdrant.py
======================
Qdrant client wrapper.

The collection is now bootstrapped with INT8 Scalar Quantization and
on-disk storage by db/setup_collection.py (or docker-compose startup).
This module focuses purely on runtime upsert + search operations.

If the collection doesn't exist yet at startup, it falls back to creating
a quantized collection automatically so the app remains self-contained in
development.
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings

log = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection()

    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """
        Create the collection if it doesn't exist yet.
        Mirrors the configuration in db/setup_collection.py so that the
        app also works without running that script manually.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            log.info("Connected to existing collection '%s'.", self.collection_name)
            return

        log.info(
            "Collection '%s' not found. Creating with INT8 quantization…",
            self.collection_name,
        )
        self.client.create_collection(
            collection_name=self.collection_name,

            # ── Vectors ────────────────────────────────────────────────
            vectors_config=models.VectorParams(
                size=768,  # SigLIP-base-patch16-224 output dim
                distance=models.Distance.COSINE,
                on_disk=True,  # mmap → low RAM footprint
            ),

            # ── INT8 Scalar Quantization ───────────────────────────────
            # Reduces per-vector storage from 3 072 bytes (768×FP32)
            # to 768 bytes (768×INT8) – a 4× RAM saving per vector.
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,   # Calibrate on 99th pct of values
                    always_ram=False,  # Keep quantized vecs on disk
                )
            ),

            # ── HNSW index ─────────────────────────────────────────────
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                on_disk=True,
            ),

            # ── Payload ────────────────────────────────────────────────
            on_disk_payload=True,
        )

        # Payload indexes for O(1) multi-tenant filtering
        for field in ("user_id", "video_id"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        log.info("Collection '%s' created with INT8 quantization.", self.collection_name)

    # ------------------------------------------------------------------
    def upsert_vectors(
        self,
        user_id: str,
        video_id: str,
        vectors: list,
        timestamps: list,
    ) -> int:
        """
        Upsert a list of embedding vectors with their metadata.
        Returns the total number of points upserted.
        """
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "user_id": user_id,
                    "video_id": video_id,
                    "timestamp": timestamps[i],
                },
            )
            for i, vec in enumerate(vectors)
        ]

        # Batch upsert in chunks of 100 (Qdrant recommendation)
        batch_size = 100
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )

        return len(points)

    # ------------------------------------------------------------------
    def search(self, user_id: str, query_vector: list, limit: int = 5) -> list:
        """
        Search for the closest vectors belonging to *user_id*.
        Rescore with the original FP32 vectors for higher accuracy when
        quantized vectors are used (Qdrant oversampling + rescore).
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
            # Oversample to compensate for quantisation recall loss,
            # then rescore against exact vectors.
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,       # Re-rank with original FP32 vectors
                    oversampling=2.0,   # Retrieve 2× candidates before rescore
                )
            ),
            limit=limit,
        ).points

        return results


qdrant_service = QdrantService()
