from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from typing import Optional
import shutil
import os
import uuid
import torch
import torch.nn.functional as F
import logging
from app.services.tasks import process_video_background, tasks_db
from app.services.qdrant import qdrant_service
from app.engine.models import SigLipEngine 
from app.core.config import settings

log = logging.getLogger(__name__)

router = APIRouter()

# Ensure temp upload folder exists
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    # Save file temporarily
    file_extension = file.filename.split(".")[-1]
    task_id = str(uuid.uuid4())
    temp_file_path = os.path.join(UPLOAD_DIR, f"{task_id}.{file_extension}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Queue background task
    tasks_db[task_id] = {"status": "queued"}
    background_tasks.add_task(process_video_background, task_id, temp_file_path, user_id)
    
    return {"task_id": task_id, "status": "queued"}

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    status = tasks_db.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@router.post("/search")
async def search_video(
    query: str = Form(...),
    user_id: str = Form(...),
    limit: int = 5
):
    siglip = SigLipEngine()
    model, processor, device = siglip.get_components()
    
    try:
        inputs = processor(text=[query], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            if attention_mask is not None:
                text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            else:
                text_features = model.get_text_features(input_ids=input_ids)

            text_features = F.normalize(text_features, p=2, dim=1)
            text_vector = text_features.cpu().numpy()[0].tolist()
            
    except Exception as e:
        log.error(f"Error embedding query: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    results = qdrant_service.search(user_id, text_vector, limit)
    
    response = []
    for res in results:
        response.append({
            "score": res.score,
            "timestamp": res.payload.get("timestamp"),
            "video_id": res.payload.get("video_id"),
            # "user_id" is redundant as we searched for it
        })
        
    return response
