from fastapi import FastAPI
from app.api import routes
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Semantic Video Search API is running"}
