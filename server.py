from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from befs.config import settings
from befs.route import middleware, v1
from befs.train import BaseMLTrainer

cors_origins = [
    "http://localhost",
    "https://localhost",
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://localhost:5000",
    "https://localhost:5000",
    "http://127.0.0.1:5000",
    "https://127.0.0.1:5000",
    "https://capstone.smccnasipit.edu.ph/befs",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # on startup
    training_classes: Dict[str, BaseMLTrainer] = {}
    app.training_classes = training_classes
    yield  # Running server
    # on shutdown

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(middleware.ProcessTimeMiddleware)

app.add_middleware(middleware.APIKeyMiddleware)

# Include API routes
app.include_router(v1.router, prefix="/api")

@app.get("/favicon.ico")
async def disable_favicon():
    return {"detail": "No favicon"}
@app.get("/")
async def homepage():
    return {"SUCCESS": "BEFS API is running"}

@app.get("/*")
async def catch_all():
    return {"ERROR": "Endpoint not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=settings.FASTAPI_SERVER_HOST, port=int(settings.FASTAPI_SERVER_PORT), reload=False)
