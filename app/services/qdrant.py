from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings
import logging
import uuid

log = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            log.info(f"Creating collection {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            # Create Payload Index for user_id and video_id for faster filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="video_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def upsert_vectors(self, user_id: str, video_id: str, vectors: list, timestamps: list):
        """
        vectors: list of lists (embeddings)
        timestamps: list of floats
        """
        points = []
        for i, vec in enumerate(vectors):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()), 
                vector=vec,
                payload={
                    "user_id": user_id,
                    "video_id": video_id,
                    "timestamp": timestamps[i]
                }
            ))
        
        # Batch upsert
        # Qdrant recommends batch size of 100-500. 
        # Since our inference batches are 32, we can just push them or aggregate.
        
        start_idx = 0
        batch_limit = 100
        while start_idx < len(points):
            chunk = points[start_idx : start_idx + batch_limit]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=chunk
            )
            start_idx += batch_limit
            
        return len(points)

    def search(self, user_id: str, query_vector: list, limit: int = 5):
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=limit
        ).points
        return search_result

qdrant_service = QdrantService()
