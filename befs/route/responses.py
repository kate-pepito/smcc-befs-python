from datetime import datetime
from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel

class TrainCreateSessionRequest(BaseModel):
    username: str
    session_key: str
    algo: Optional[Literal["Logistic Regression", "XGBoost Classifier"]] = None
    train_token: Optional[str]

class TrainCreateSessionPost(BaseModel):
    username: str
    session_key: str
    algo: Optional[Literal["Logistic Regression", "XGBoost Classifier"]] = None
    token: str

class TrainCreateSessionResponse(BaseModel):
    session_token: Optional[str] = None

class TrainDestroySessionResponse(BaseModel):
    success: bool
    detail: str

class SessionValidateRequest(BaseModel):
    username: str
    session_key: str
    token: Optional[str] = None
    algo: Optional[str] = None

class InvalidateSessionRequest(BaseModel):
    token: str

class SessionValidateResponse(BaseModel):
    valid: bool

class TrainSessionsGet(BaseModel):
    data: List[str] = []

class TrainSessionsResponse(BaseModel):
    data: List[List[str]] = []

class MLModelMetadata(BaseModel):
    algo: Literal["Logistic Regression", "XGBoost Classifier"]
    size: float
    filename: str
    file_extension: str
    filepath: str
    accuracy: float
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class FileModelResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    filepath: Optional[str] = None

class DatasetMetadata(BaseModel):
    filename: str
    size: Union[str, int, float]
    filepath: str
    rows: Optional[int] = None
    columns: Optional[int] = None

class DatasetRemoveFile(BaseModel):
    dataset: str

class ModelRemoveFile(BaseModel):
    model: str

class TrainingStatesResponse(BaseModel):
    connection: Literal["connected", "disconnected"]
    status: Literal["idle", "training", "completed", "error"]
    started_at: datetime
    token: str
    session_id: str
    username: str
    progress: float
    ended_at: Optional[datetime] = None
    algo: Literal["Logistic Regression", "XGBoost Classifier"]
    training_start_time: Optional[float] = None
    training_end_time: Optional[float] = None
    last_training_time: Optional[float] = None
    dataset: Optional[DatasetMetadata] = None
    column_names: Optional[List[str]] = None
    features: List[str] = []
    target: List[str] = []
    valid_hyperparameters: List[str] = []
    hyperparameters: dict
    test_size: float
    random_state: int
    scaler: dict
    model: Optional[MLModelMetadata] = None
    metrics: Optional[Any] = None
    error: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            Optional[datetime]: lambda dt: dt.isoformat() if dt is not None else None,
        }

class CommandRequest(BaseModel):
    action: str
    data: Optional[Union[str,dict,list,float,int]] = None

class SaveMLModelResponse(BaseModel):
    state: Literal["save_start", "save_end", "save_failed"]

class UpdateStateResponse(BaseModel):
    success: bool

class CreateSessionTrainResponse(BaseModel):
    token: Optional[str] = None
    detail: Optional[str] = None
    