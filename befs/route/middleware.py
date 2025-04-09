from ast import Dict
import time
from typing import Optional
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from befs.config import settings
from befs.train import BaseMLTrainer


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # api_key = request.get("api_key")
        uripath: str = request.get("path")
        if uripath.startswith("/api"):
            api_key = request.query_params.get("api_key")
            if api_key != settings.API_KEY:
                return JSONResponse({"detail": "Invalid API Key"}, status_code=403)
        response = await call_next(request)
        return response


    
async def check_api_key(websocket: WebSocket, api_key: str) -> bool:
    api_key = websocket.query_params.get("api_key")
    if api_key != settings.API_KEY:
        await websocket.close()
        return False
    return True

def get_trainer_class(websocket: WebSocket, session_token: str) -> Optional[BaseMLTrainer]:
    trainer_classes: Dict[str, BaseMLTrainer] = websocket.app.training_classes
    return trainer_classes[session_token] if session_token in trainer_classes.keys() else None

