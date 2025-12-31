import cv2
import queue
import threading
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from app.engine.models import SigLipEngine
from app.core.config import settings

class VideoFrameProducer(threading.Thread):
    def __init__(self, video_path: str, frame_queue: queue.Queue, batch_size=32):
        super().__init__()
        self.video_path = video_path
        self.frame_queue = frame_queue
        # self.stop_event = stop_event
        self.batch_size = batch_size
        self.daemon = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video: {self.video_path}")
            self.frame_queue.put(None) # Sentinel
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        
        # We want approx 1 frame per second. The original code supported 'fast', 'accurate', '1fps'.
        # For simplicity and efficiency in this refactor, let's target 1 FPS extraction by skipping frames.
        # If fps is 30, we pick every 30th frame.
        
        skip_frames = int(fps) if fps > 0 else 30 
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % skip_frames == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                timestamp = frame_idx / fps if fps > 0 else 0.0
                
                self.frame_queue.put({
                    "image": pil_image,
                    "timestamp": timestamp,
                    "frame_idx": frame_idx
                })
            
            frame_idx += 1
        
        cap.release()
        self.frame_queue.put(None) # End signal


class InferenceConsumer:
    def __init__(self, frame_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.siglip = SigLipEngine()
        self.model, self.processor, self.device = self.siglip.get_components()

    def process_video(self):
        """
        Consumes frames from queue, batches them, and runs inference.
        Returns a list of dicts: [{"embedding": [...], "timestamp": float}, ...]
        """
        results = []
        batch = []
        
        while True:
            item = self.frame_queue.get()
            
            if item is None:
                # Process remaining batch
                if batch:
                    results.extend(self._run_batch(batch))
                break
            
            batch.append(item)
            
            if len(batch) >= 32: 
                results.extend(self._run_batch(batch))
                batch = []
                
        return results

    def _run_batch(self, batch):
        images = [item["image"] for item in batch]
        timestamps = [item["timestamp"] for item in batch]
        
        # Preprocess
        inputs = self.processor(images=images, return_tensors="pt", padding="max_length") 
        
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            # Use get_image_features for SiglipModel
            embeddings = self.model.get_image_features(pixel_values=pixel_values)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings_np = embeddings.cpu().numpy().astype("float32")
            
        batch_results = []
        for i, emb in enumerate(embeddings_np):
            batch_results.append({
                "vector": emb.tolist(),
                "timestamp": timestamps[i]
            })
            
        return batch_results
