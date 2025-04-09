
import asyncio
from datetime import datetime, timezone
import json
import secrets
import time
from typing import Any, List, Literal, Optional, Union
from fastapi.websockets import WebSocketState
import numpy as np
import pandas as pd
from fastapi import WebSocket
from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, accuracy_score, roc_curve
from befs.http_request import end_session, get_dataset_contents, invalidate_train_session_token, remove_dataset_file, remove_model_file, update_training_state, upload_model
from sklearn.pipeline import Pipeline
from befs.route.responses import CommandRequest, DatasetMetadata, MLModelMetadata, SaveMLModelResponse, TrainCreateSessionRequest, TrainingStatesResponse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from xgboost import XGBClassifier
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
)
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

class BaseMLTrainer:
    def __init__(self, session_id: str, username: str, token: str, valid_hyperparameters: List[str], algo: Literal["Logistic Regression", "XGBoost Classifier"]):
        self.valid_hyperparameters = valid_hyperparameters
        self.algo = algo
        self.session_id = session_id
        self.username = username
        self.token = token
        self.model = None
        self.saved_model = None
        self.websocket = None
        self.dataset = None
        self.features = []
        self.target = []
        self.hyperparameters = {}
        self.test_size = 0.2
        self.random_state = 42
        self.scaler_class = {}
        self.save_model_full_path = ""
        self.state = TrainingStatesResponse(
            connection="connected",
            status="idle",
            progress=0.0,
            algo=algo,
            session_id=session_id,
            username=username,
            token=token,
            scaler={},
            valid_hyperparameters=valid_hyperparameters,
            hyperparameters={},
            column_names=[],
            features=[],
            target=[],
            test_size=0.2,
            random_state=42,
            started_at=datetime.now(timezone.utc)
        )
    
    def connect(self, websocket: WebSocket):
        self.websocket = websocket
        self.state.connection = "connected"

    async def update_state(self):
        await update_training_state(self.token, self.state)
        await self.send_updates()

    async def send_updates(self):
        if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
            state = self.state.model_dump_json()
            try:
                state = json.loads(state)
            except Exception as e:
                print("send_updates:", str(e))

            await self.websocket.send_json(state)

    async def set_dataset(self, dataset: Optional[Union[list, pd.DataFrame]] = None, metadata: Optional[DatasetMetadata] = None):
        try:
            if dataset is None or metadata is None:
                self.dataset = None
                self.state.dataset = None
                self.state.column_names = []
                self.features = []
                self.target = []
                self.state.features = []
                self.state.target = []
                raise Exception("[expected exception]: removed dataset")
            else:
                if isinstance(dataset, pd.DataFrame):
                    self.dataset = dataset
                elif isinstance(dataset, (list, tuple)) and len(dataset) > 0 and isinstance(dataset[0], dict):
                    self.dataset = pd.DataFrame(dataset)
                elif isinstance(dataset, (list, tuple)) and len(dataset) > 0:
                    column_names = [f"column_{i+1}" for i in range(len(dataset[0]))] if dataset and isinstance(dataset[0], (list, tuple)) else ["value"]
                    dataset = {col: [row[i] for row in dataset] for i, col in enumerate(column_names)} if dataset and isinstance(dataset[0], (list, tuple)) else {"value": dataset}
                    self.dataset = pd.DataFrame(dataset)
                else:
                    raise Exception("Invalid Dataset!")
                self.state.column_names = list(self.dataset.columns)
                old_dataset = self.state.dataset
                self.state.dataset = DatasetMetadata(**metadata.model_dump(exclude=['columns', 'rows']), columns=len(self.dataset.columns), rows=int(self.dataset.shape[0]))
                if old_dataset is not None:
                    filename = old_dataset.filename
                    await remove_dataset_file(filename)
                
        except Exception as e:
            print("exception here?", e)
            self.state.status = "error"
            self.state.error = f"set_dataset: {str(e)}"
            await self.update_state()
            self.state.status = "idle"
        finally:
            await self.update_state()

    async def set_features(self, *features):
        self.features = list(filter(lambda ft: ft in self.state.column_names, features))
        self.state.features = self.features
        await self.update_state()

    async def set_target(self, *target):
        self.target = list(filter(lambda tg: tg in self.state.column_names and tg not in self.features, target))
        self.state.target = self.target
        await self.update_state()

    async def set_hyperparameters(self, **hyperparameters):
        hi = hyperparameters.items()
        hyperparameter_items = []
        for k,v in hi:
            if type(v) is str:
                try:
                    if "." in v:
                        cv = float(v)
                    else:
                        cv = int(v)
                except ValueError:
                    try:
                        cv = json.loads(v)
                    except json.JSONDecodeError:
                        cv = v
            else:
                cv = v
            hyperparameter_items.append((k,cv))
        my_hyperparameters = {
            key: value for key, value in hyperparameter_items if key in self.valid_hyperparameters and value is not None and value != ""
        }
        self.hyperparameters = my_hyperparameters
        self.state.hyperparameters = self.hyperparameters
        await self.update_state()
    
    async def set_test_size(self, test_size: float):
        self.test_size = test_size
        self.state.test_size = self.test_size
        await self.update_state()
    
    async def set_random_state(self, random_state: float):
        self.random_state = random_state
        self.state.random_state = self.random_state
        await self.update_state()

    async def _train(self, X_train, y_train, X_test, y_test) -> tuple:
        pass

    async def train(self) -> bool:
        if self.dataset is not None and len(self.features) > 0 and len(self.target) > 0 and self.state.status != "training":
            self.state.status = "training"
            self.state.training_start_time = time.time()
            self.state.training_end_time = None
            self.state.progress = 30
            await self.update_state()

            try:
                X = self.dataset[self.features].values  # Features
                y = self.dataset[self.target].values   # Target

                # Check if y contains string labels, then encode
                if y.dtype == 'object' or y.dtype.name == 'category':
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
                self.state.progress = 50
                await self.update_state()

                await self._train(X_train, y_train, X_test, y_test)

                self.state.progress = 90
                await self.update_state()

                # Training complete
                self.state.status = "completed"
                self.state.training_end_time = time.time()
                self.state.last_training_time = self.state.training_end_time - self.state.training_start_time
                self.state.progress = 100
                await self.update_state()

            except Exception as e:
                self.state.status = "error"
                self.state.error = str(e)
                self.state.progress = 0
                await self.update_state()
                self.state.status = "idle"
                await self.update_state()
            finally:
                return True
        return False
    
    async def save_model(self):
        if self.state.status == "completed":
            if hasattr(self, "_save_model") and callable(self._save_model):
                serialized_model = await self._save_model()
                self.saved_model = serialized_model
                filename = secrets.token_hex(12)
                file_ext = ".onnx"
                filepath = "/inference/"
                self.state.model = MLModelMetadata(filename=filename, file_extension=file_ext, filepath=filepath, algo=self.algo, created_at=datetime.now(timezone.utc), size=len(serialized_model), accuracy=float(self.state.metrics["accuracy"]))
            else:
                initial_type = [("input", FloatTensorType([None, len(self.features)]))]
                onx = convert_sklearn(self.model, initial_types=initial_type)
                serialized_model = onx.SerializeToString()
                self.saved_model = serialized_model
                filename = secrets.token_hex(12)
                file_ext = ".onnx"
                filepath = "/inference/"
                self.state.model = MLModelMetadata(filename=filename, file_extension=file_ext, filepath=filepath, algo=self.algo, created_at=datetime.now(timezone.utc), size=len(serialized_model), accuracy=self.state.metrics["accuracy"])
            await self.update_state()
            
    async def send_model(self, customfilepath: Optional[str] = None):
        if self.saved_model is not None and self.state.model is not None:
            try:
                state_model = MLModelMetadata(**self.state.model.model_dump(exclude=['filepath']), filepath=customfilepath) if customfilepath is not None else self.state.model
                print("uploading model..", state_model)
                resp = await upload_model(self.saved_model, state_model)
                print("result:", resp)
                if resp is not None and resp.success:
                    self.save_model_full_path = resp.filepath
                    resp = SaveMLModelResponse(state="save_end").model_dump()
                    await self.websocket.send_json(resp)
                else:
                    raise Exception("Error")
            except Exception as e:
                print("ERROR ON SAVE:", str(e), e.__traceback__.tb_lineno)
                resp = SaveMLModelResponse(state="save_failed").model_dump()
                await self.websocket.send_json(resp)
    
    async def remove_model(self, model_path: Optional[str] = None):
        if self.saved_model is not None and self.state.model is not None:
            try:
                model_path = f"{model_path}{self.state.model.filename}{self.state.model.file_extension}" if model_path is not None else f"{self.state.model.filepath}{self.state.model.filename}{self.state.model.file_extension}"
                await remove_model_file(model_path)
            except Exception as e:
                print("failed remove model:", str(e))
                await self.update_state()

    async def run_command(self, command: CommandRequest):
        if command.action == "get_updates":
            await self.send_updates()
        elif command.action == "set_dataset":
            if command.data is not None:
                datasetmetadata = DatasetMetadata(**command.data)
                dataset = await get_dataset_contents(datasetmetadata)
                await self.set_dataset(dataset, datasetmetadata)
            else:
                await self.set_dataset(None, None)
        elif command.action == "set_features":
            features = command.data
            await self.set_features(*features)
        elif command.action == "set_target":
            target = command.data
            await self.set_target(*target)
        elif command.action == "set_test_size":
            test_size = None if command.data == "" or command.data is None or not command.data else float(command.data)
            await self.set_test_size(test_size)
        elif command.action == "set_random_state":
            random_state = None if command.data == "" or command.data is None or not command.data else int(command.data)
            await self.set_random_state(random_state)
        elif command.action == "set_hyperparameters":
            hyperparameters = None if command.data == "" or command.data is None or not command.data else command.data
            await self.set_hyperparameters(**hyperparameters)
        elif command.action == "save_model":
            await self.train()
            await self.save_model()
        elif command.action == "upload_model":
            filepath = command.data["filepath"] if command.data is not None else None
            await self.send_model(filepath)
        elif command.action == "remove_model":
            filepath = command.data["filepath"] if command.data is not None else None
            await self.remove_model(filepath)
        elif command.action == "end_session":
            await self.set_dataset(None, None)
            self.state.ended_at = datetime.now(timezone.utc)
            await self.update_state()
            await end_session(TrainCreateSessionRequest(username=self.username, session_key=self.session_id, algo=self.algo, train_token=self.token))
        else:
            await self.send_updates()

    async def websocket_loop(self):
        if self.websocket:
            await self.update_state()
            while True:
                try:
                    message = await self.websocket.receive_json()
                    await self.run_command(CommandRequest(**message))
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error: {e}")
                    break  # Break loop on error or disconnect

    async def disconnect(self):
        self.state.connection = "disconnected"
        await self.update_state()
        self.websocket = None
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await invalidate_train_session_token(self.token)

    def __del__(self):
        loop = asyncio.get_event_loop()
        loop.create_task(invalidate_train_session_token(str(self.token)))

class LogisticRegressionTrainer(BaseMLTrainer):
    def __init__(self, session_id: str, username: str, token: str):
        super().__init__(session_id, username, token, valid_hyperparameters = [
            "penalty", "dual", "tol", "C", "fit_intercept", "intercept_scaling",
            "class_weight", "random_state", "solver", "max_iter", "multi_class",
            "verbose", "warm_start", "n_jobs", "l1_ratio"
        ], algo = "Logistic Regression")
    
    async def _train(self, X_train, y_train, X_test, y_test):
        self.scaler_class = StandardScaler()
        self.state.progress = 60
        await self.update_state()
        self.model: Pipeline = Pipeline([
            #("scaler", self.scaler_class),
            ('imputer', SimpleImputer(strategy='mean')),
            ("logreg", LogisticRegression(**self.hyperparameters))
        ])
        self.model.fit(X_train, y_train.ravel())
        self.state.progress = 70
        await self.update_state()
        # scaler_steps: StandardScaler = self.model.named_steps['scaler']
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else np.zeros_like(y_pred)  # For ROC & PR curves
        self.state.metrics = {
            "classification_report": classification_report(y_test.ravel(), y_pred, output_dict=True),
            "accuracy": float(accuracy_score(y_test.ravel(), y_pred)),
            "precision": float(precision_score(y_test.ravel(), y_pred, average="weighted")),
            "recall": float(recall_score(y_test.ravel(), y_pred, average="weighted")),
            "f1_score": float(f1_score(y_test.ravel(), y_pred, average="weighted")),
            "confusion_matrix": confusion_matrix(y_test.ravel(), y_pred).tolist(),
        }
        if len(np.unique(y_test)) == 2:  # Check if binary classification
            fpr, tpr, _ = roc_curve(y_test.ravel(), y_proba)
            self.state.metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            self.state.metrics["roc_auc"] = float(auc(fpr, tpr))  # ROC AUC Score

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test.ravel(), y_proba)
            self.state.metrics["precision_recall_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
            self.state.metrics["pr_auc"] = float(auc(recall, precision))  # PR AUC Score

        self.state.scaler =  {
            # "mean": scaler_steps.mean_.tolist(),
            # "scale": scaler_steps.scale_.tolist()
        }
        self.state.progress = 80
        await self.update_state()
    
    async def _save_model(self) -> Any:
        xgb_onnx = convert_sklearn(
            self.model,
            "pipeline_logreg",
            [("input", FloatTensorType([None, len(self.features)]))],
            target_opset={"": 17, "ai.onnx.ml": 3},
            options={id(self.model): {"zipmap": "columns"}}  # Enable probability scores
        )

        return xgb_onnx.SerializeToString()

class XGBClassifierTrainer(BaseMLTrainer):
    def __init__(self, session_id: str, username: str, token: str):
        super().__init__(session_id, username, token, valid_hyperparameters = [
           "n_estimators", "max_depth", "learning_rate", "verbosity", "objective",
            "booster", "tree_method", "gamma", "min_child_weight", "max_delta_step",
            "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode",
            "reg_alpha", "reg_lambda", "scale_pos_weight", "base_score", "random_state",
            "missing", "importance_type", "grow_policy", "max_leaves", "max_bin",
            "eval_metric", "early_stopping_rounds", "use_label_encoder"
        ], algo = "XGBoost Classifier")

    async def _train(self, X_train, y_train, X_test, y_test):
        self.scaler_class = StandardScaler()
        self.state.progress = 60
        await self.update_state()
        self.model: Pipeline = Pipeline([
            # ("scaler", self.scaler_class),
            ('imputer', SimpleImputer(strategy='mean')),
            ("xgb", XGBClassifier(**self.hyperparameters))
        ])
        self.model.fit(X_train, y_train.ravel())
        self.state.progress = 70
        await self.update_state()
        # scaler_steps: StandardScaler = self.model.named_steps['scaler']
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else np.zeros_like(y_pred)  # For ROC & PR curves
        self.state.metrics = {
            "classification_report": str(classification_report(y_test.ravel(), y_pred, output_dict=True)),
            "accuracy": float(accuracy_score(y_test.ravel(), y_pred)),
            "precision": float(precision_score(y_test.ravel(), y_pred, average="weighted")),
            "recall": float(recall_score(y_test.ravel(), y_pred, average="weighted")),
            "f1_score": float(f1_score(y_test.ravel(), y_pred, average="weighted")),
            "confusion_matrix": confusion_matrix(y_test.ravel(), y_pred).tolist(),
        }
        if len(np.unique(y_test)) == 2:  # Check if binary classification
            fpr, tpr, _ = roc_curve(y_test.ravel(), y_proba)
            self.state.metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            self.state.metrics["roc_auc"] = float(auc(fpr, tpr))  # ROC AUC Score

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test.ravel(), y_proba)
            self.state.metrics["precision_recall_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
            self.state.metrics["pr_auc"] = float(auc(recall, precision))  # PR AUC Score

        self.state.scaler =  {
            # "mean": scaler_steps.mean_.tolist(),
            # "scale": scaler_steps.scale_.tolist()
        }
        self.state.progress = 80
        await self.update_state()

    async def _save_model(self) -> Any:
        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        xgb_onnx = convert_sklearn(
            self.model,
            "pipeline_xgboost",
            [("input", FloatTensorType([None, len(self.features)]))],
            target_opset={"": 17, "ai.onnx.ml": 3},
            options={id(self.model): {"zipmap": "columns"}}  # Enable probability scores
        )
        return xgb_onnx.SerializeToString()