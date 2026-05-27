# 🔍 Semantic Video Search Engine 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=flat&logo=fastapi)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker)

> **"Find the exact moment a car crashes"** or **"Show me someone cooking pasta"**  across hours of videos, in milliseconds.

A high-throughput, horizontally scalable video search engine powered by **Google's SigLIP** (Sigmoid Loss for Language Image Pre-Training) and **Qdrant**. Designed for production environment with asynchronous processing, producer-consumer pipelines, and multi-tenancy.

---

## Performance Markers

| Benchmark | Result | Environment |
| :--- | :--- | :--- |
| **Throughput** | ~21 Seconds for 100 mins of video | T4 GPU (Google Colab) |
| **Inference Latency** | < 500ms per batch (32 frames) | Google Colab  |
| **Search Speed** | < 300ms  | Qdrant (HNSW Index) |

---

## System Architecture

This project has evolved from a simple script into a robust microservice architecture.

```mermaid
graph TD
    User[User / Client] -->|Upload Video| API[FastAPI Backend]
    API -->|Enqueue Task| TaskQueue[Background Tasks]
    
    subgraph "Ingestion Pipeline (Producer-Consumer)"
        TaskQueue -->|Spawn| VLoad[Video Loader Thread]
        VLoad -->|Stream Frames| MemBuf["Memory Buffer (Queue)"]
        MemBuf -->|Batch Fetch| Inference[SigLIP Inference Engine]
    end
    
    Inference -->|Generate Vectors| Embeddings[Multimodal Embeddings]
    Embeddings -->|Upsert| Qdrant[Qdrant Vector DB]
    
    User -->|Search Query| API
    API -->|Query Vector| Qdrant
    Qdrant -->|Ranked Results| API

```

### Key Engineering Highlights

* **Multi Threaded Pipeline :** Decoupled CPU-bound video decoding from GPU bound model inference using thread safe memory queues. This design mitigates GPU starvation

* **Zero Copy In Memory Streaming :** Extracted video frames are streamed directly through RAM buffers into the inference batch engine without intermediate local disk writes, completely eliminating disk I/O bottlenecks and reducing hardware wear.

* **Non Blocking Asynchronous API :** Implemented utilizing FastAPI and Python's asyncio to effortlessly manage high-concurrency connections during long-running background extraction worker processes. ( branch not yet PRed )

---



## Tech Stack

*   **Core Backend**: Python 3.10, FastAPI
*   **AI / ML**: PyTorch, Transformers , **Google SigLIP** 
*   **Computer Vision**: OpenCV , ffmpeg
*   **Database**: Qdrant (Vector Store)
*   **Infrastructure**: Docker

---

## Installation & Usage

### 1. Clone & Setup
```bash
git clone https://github.com/Aniket-16-S/Semantic_Video_Search.git
cd Semantic_Video_Search
pip install -r requirements.txt
```

### 2. Run the Vector Database (Qdrant)
You need a running Qdrant instance. The easiest way is via Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```
*Or use Qdrant Cloud for a managed instance.*

### 3. Start the API Server
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.

### 4. Interactive Documentation
Go to `http://localhost:8000/docs` to test the endpoints interactively:
*   **POST /upload**: Upload video files for processing.
*   **POST /search**: Search your indexed videos using text.
*   **DELETE /reset**: Clear the index.

---

## Roadmap & Future Improvements

- [x] **v1.0**: Core Script (SigLip + Qdrant )
- [x] **v2.0**: FastAPI Backend + Producer-Consumer Pipeline + Qdrant (about to be released)
- [ ] **v3.0**: MoonDream, support for android Devices.
- [ ] **v4.0**: Distributed Worker Nodes (Celery/Redis) for horizontal scaling
- [ ] **Frontend**: Dashboard for video managment

---

*Engineered by [Aniket-16-S](https://github.com/Aniket-16-S)*
