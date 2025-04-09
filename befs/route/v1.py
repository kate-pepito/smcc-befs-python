from datetime import datetime
import secrets
from typing import Dict
from fastapi import APIRouter,  Request, WebSocket, WebSocketDisconnect
from befs.http_request import create_train_session_api, get_train_session, get_train_sessions, invalidate_train_session, invalidate_train_session_token, validate_train_session

from befs.route import middleware
from befs.route.responses import InvalidateSessionRequest, SessionValidateRequest, SessionValidateResponse, TrainCreateSessionPost, TrainCreateSessionRequest, TrainCreateSessionResponse, TrainDestroySessionResponse, TrainSessionsResponse, TrainingStatesResponse
from befs.train import BaseMLTrainer, LogisticRegressionTrainer, XGBClassifierTrainer

router = APIRouter()
router.prefix = "/v1"

@router.post("/train/create", response_model=TrainCreateSessionResponse)
async def create_training_session(data: SessionValidateRequest, request: Request):
    session_token = None
    try:
        resp = await validate_train_session(data)
        if resp.valid:
            session_token = data.token
        else:
            session_token = str(secrets.token_hex(16))
            # Initialize training state
            training_classes: Dict[str, BaseMLTrainer] = request.app.training_classes
            TrainerClass = LogisticRegressionTrainer if data.algo == "Logistic Regression" else XGBClassifierTrainer
            training_classes[str(session_token)] = TrainerClass(
                session_id=data.session_key,
                username=data.username,
                token=str(session_token)
            )
            create_body = TrainCreateSessionPost(**data.model_dump(exclude='token'), token=str(session_token))
            await create_train_session_api(create_body)
    except Exception as e:
        print(e)
        session_token = None
    finally:
        return TrainCreateSessionResponse(session_token=str(session_token) if session_token is not None else None)

@router.get("/validate/session", response_model=SessionValidateResponse)
async def validate_session(request: Request):
    is_valid = False
    try:
        params = request.query_params.items()
        param_dict = {p[0]:p[1] for p in params}
        data = SessionValidateRequest(**param_dict)
        resp = await validate_train_session(data)
        if not resp or not resp.valid:
            raise Exception("[expected exception] Session Invalid")
        training_classes: Dict[str, BaseMLTrainer] = request.app.training_classes
        is_valid = data.token in training_classes.keys()
        if not is_valid:
            await invalidate_train_session(data)
    except Exception as e:
        print(e)
        is_valid = False
    finally:
        return SessionValidateResponse(valid=is_valid)

@router.get("/train/sessions", response_model=TrainSessionsResponse)
async def get_training_sessions(request: Request):
    try:
        resp = await get_train_sessions()
        keys = resp.data if resp is not None else []
        training_classes: Dict[str, BaseMLTrainer] = request.app.training_classes
        data = []
        for tck in keys:
            if tck in training_classes.keys():
                username = training_classes[str(tck)].username
                session_key = training_classes[str(tck)].session_id
                algo = training_classes[str(tck)].algo
                started = str(training_classes[str(tck)].state.started_at)
                token = training_classes[str(tck)].token
                data.append([username, session_key, algo, started, token])
            else:
                await invalidate_train_session_token(InvalidateSessionRequest(token=tck))
        data.sort(key=lambda x: datetime.fromisoformat(x[3]))
    except Exception as e:
        print(e)
        data = []
    finally:
        return TrainSessionsResponse(data=data)

@router.post("/train/destroy", response_model=TrainDestroySessionResponse)
async def destroy_training_session(data: TrainCreateSessionRequest, request: Request):
    detail = ""
    success = False
    try:
        resp = await get_train_session(data)
        session_token = resp.session_token if resp is not None else None
        if session_token is not None:
            await invalidate_train_session_token(InvalidateSessionRequest(token=session_token))
            training_classes: Dict[str, LogisticRegressionTrainer] = request.app.training_classes
            if str(session_token) in training_classes.keys():
                del training_classes[str(session_token)]
                detail = f"{str(session_token)} session deleted"
                success = True
            else:
                raise Exception("No Training Session Found")
        else:
            raise Exception("No Training Session Found")
    except Exception as e:
        print(e)
        detail = str(e)
        success = False
    finally:
        return TrainDestroySessionResponse(success=success, detail=detail)

@router.websocket("/train")
async def websocket_endpoint(websocket: WebSocket, api_key: str, token: str):
    try:
        # api_key = websocket.query_params.get("api_key")
        # session_token = websocket.query_params.get("token")
        session_token = token
        valid_api_key = await middleware.check_api_key(websocket, api_key)
        if not valid_api_key:
            return

        trainer = middleware.get_trainer_class(websocket, session_token)
        if trainer is None:
            resp = TrainingStatesResponse(connection="disconnected", state="error", progress=0.0, error="Training Session not yet initiated").model_dump_json()
            await websocket.send_text(resp)
            await websocket.close()
        await websocket.accept()
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {str(e)}")
        return
    except Exception as e:
        print(f"Error Websocket Connection: {str(e)}")

    try:
        trainer.connect(websocket)
        await trainer.websocket_loop()
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {trainer.session_id}")
        await trainer.disconnect()
    except Exception as e:
        trainer.state.status = "error"
        trainer.state.error = str(e)
        await trainer.update_state()