import logging
import uuid
from queue import Queue
import threading
from app.engine.pipeline import VideoFrameProducer, InferenceConsumer
from app.services.qdrant import qdrant_service

log = logging.getLogger(__name__)

# Simple in-memory task:status storage.
tasks_db = {} 

def update_task_status(task_id, status, error=None):
    tasks_db[task_id] = {"status": status, "error": error}
    log.info(f"Task {task_id} updated to {status}")

def process_video_background(task_id: str, video_path: str, user_id: str):
    try:
        update_task_status(task_id, "processing")
        
        frame_queue = Queue(maxsize=128) # Limit memory usage
        
        producer = VideoFrameProducer(video_path, frame_queue)
        consumer = InferenceConsumer(frame_queue)
        
        # Start Producer
        producer.start()
        
        # Run Consumer (Inference)
        # This blocks until producer finishes and queue is empty
        # We process in chunks to stream to DB? 
        # consumer.process_video() returns ALL results. 
        # use the list for now as video length is likely manageable
        
        embeddings = consumer.process_video()
        
        # Upsert to Qdrant
        video_id = str(uuid.uuid4())
        vectors = [e["vector"] for e in embeddings]
        timestamps = [e["timestamp"] for e in embeddings]
        
        if vectors:
            qdrant_service.upsert_vectors(user_id, video_id, vectors, timestamps)
            update_task_status(task_id, "completed")
        else:
            update_task_status(task_id, "failed", "No embeddings generated")
            
    except Exception as e:
        log.error(f"Task {task_id} failed: {e}")
        update_task_status(task_id, "failed", str(e))
