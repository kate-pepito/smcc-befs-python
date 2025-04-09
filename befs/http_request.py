import io
import json

import httpx
import pandas as pd
from typing import Union

from pydantic import BaseModel

from befs.config import settings
from befs.route.responses import CreateSessionTrainResponse, DatasetMetadata, DatasetRemoveFile,  FileModelResponse, InvalidateSessionRequest, MLModelMetadata, ModelRemoveFile, SessionValidateRequest, SessionValidateResponse, TrainCreateSessionPost, TrainCreateSessionRequest, TrainCreateSessionResponse, TrainDestroySessionResponse, TrainSessionsGet, TrainingStatesResponse, UpdateStateResponse

class ApiUrls:
    get_train_session = f"{settings.MAIN_BASE_URL}/api/get_train_session?api_key={settings.API_KEY}"
    create_train_session_api = f"{settings.MAIN_BASE_URL}/api/create_train_session?api_key={settings.API_KEY}"
    validate_train_session = f"{settings.MAIN_BASE_URL}/api/validate_session?api_key={settings.API_KEY}"
    invalidate_train_session = f"{settings.MAIN_BASE_URL}/api/invalidate_session?api_key={settings.API_KEY}"
    invalidate_train_session_token = f"{settings.MAIN_BASE_URL}/api/invalidate_session_token?api_key={settings.API_KEY}"
    get_train_sessions = f"{settings.MAIN_BASE_URL}/api/get_all_sessions?api_key={settings.API_KEY}"
    upload_model = f"{settings.MAIN_BASE_URL}/api/model_upload?api_key={settings.API_KEY}"
    remove_dataset= f"{settings.MAIN_BASE_URL}/api/remove_dataset?api_key={settings.API_KEY}"
    remove_model= f"{settings.MAIN_BASE_URL}/api/remove_model?api_key={settings.API_KEY}"
    end_session= f"/api/v1/train/destroy?api_key={settings.API_KEY}"
    def update_training_state(self, token: str):
        return f"{settings.MAIN_BASE_URL}/api/train_update?api_key={settings.API_KEY}&token={token}"
    
    def __str__(self):
        attrs = {k: v for k, v in self.__class__.__dict__.items() if not callable(v) and not k.startswith("__")}
        return "\n".join(f"{key}: {value}" for key, value in attrs.items())

apiUrls = ApiUrls()


async def http_get(url: str, params: dict = {}):
    headers = {"Accept": "application/json"}
    async with httpx.AsyncClient(verify=False) as client:
        urlsplit = url.split("?")
        more_params = urlsplit[1] if len(urlsplit) > 1 else ""
        more_params = more_params.split("&") if more_params != "" else []
        more_params = {p.split("=")[0]:p.split("=")[1] for p in more_params}
        params = {**more_params, **params}
        response = await client.get(urlsplit[0], params=params, headers=headers)
        return response.json()

async def http_get_raw(url: str, params: dict = {}):
    headers = {"Accept": "application/json"}
    async with httpx.AsyncClient(verify=False) as client:
        urlsplit = url.split("?")
        more_params = urlsplit[1] if len(urlsplit) > 1 else ""
        more_params = more_params.split("&") if more_params != "" else []
        more_params = {p.split("=")[0]:p.split("=")[1] for p in more_params}
        params = {**more_params, **params}
        response = await client.get(urlsplit[0], params=params, headers=headers)
        return response.text

async def http_post_json(url: str, data: BaseModel):
    async with httpx.AsyncClient(verify=False) as client:
        data = data.model_dump_json()
        jsonData = json.loads(data)
        response = await client.post(url, json=jsonData)
        return response.json()

async def get_train_session(data: TrainCreateSessionRequest) -> TrainCreateSessionResponse:
    respDict = await http_get(apiUrls.get_train_session, data.model_dump())
    return TrainCreateSessionResponse(**respDict)

async def create_train_session_api(data: TrainCreateSessionPost) -> CreateSessionTrainResponse:
    respDict = await http_post_json(apiUrls.create_train_session_api, data)
    return CreateSessionTrainResponse(**respDict)


async def validate_train_session(data: SessionValidateRequest) -> SessionValidateResponse:
    respDict = await http_get(apiUrls.validate_train_session, data.model_dump())
    return SessionValidateResponse(**respDict)

async def invalidate_train_session(data: SessionValidateRequest) -> TrainDestroySessionResponse:
    respDict = await http_post_json(apiUrls.invalidate_train_session, data)
    return TrainDestroySessionResponse(**respDict)

async def invalidate_train_session_token(data: InvalidateSessionRequest) -> TrainDestroySessionResponse:
    respDict = await http_post_json(apiUrls.invalidate_train_session_token, data)
    return TrainDestroySessionResponse(**respDict)

async def get_train_sessions() -> TrainSessionsGet:
    respDict = await http_get(apiUrls.get_train_sessions)
    return TrainSessionsGet(**respDict)  

async def upload_model(onnx_model: bytes, metadata: MLModelMetadata) -> FileModelResponse:
    file_data = onnx_model

    files = { "inference": (f"{metadata.filename}{metadata.file_extension}", file_data, "application/octet-stream") }
    
    async with httpx.AsyncClient(verify=False) as client:
        metadata = metadata.model_dump_json()
        metadata = json.loads(metadata)
        response = await client.post(apiUrls.upload_model, data=metadata, files=files)
        respDict = response.json()
        return FileModelResponse(**respDict)
    return FileModelResponse(success=False, error="Failed to upload model")

async def update_training_state(token: str, state: TrainingStatesResponse) -> UpdateStateResponse:
    respDict = await http_post_json(apiUrls.update_training_state(token), state)
    return UpdateStateResponse(**respDict)

async def get_dataset_contents(metadata: DatasetMetadata) -> Union[list, pd.DataFrame]:
    respDict = await http_get_raw(f"{settings.MAIN_BASE_URL}{metadata.filepath}")
    return json.loads(respDict) if respDict.startswith("[") and respDict.endswith("]") else pd.read_csv(io.StringIO(respDict))

async def remove_dataset_file(filename: str):
    await http_post_json(apiUrls.remove_dataset, DatasetRemoveFile(dataset=filename))

async def remove_model_file(filename: str):
    await http_post_json(apiUrls.remove_model, ModelRemoveFile(model=filename))

async def end_session(session: TrainCreateSessionRequest):
    await http_post_json(apiUrls.end_session, session)