import os
import secrets

from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv(".env") or find_dotenv(".env.development") or find_dotenv(".env.production")

if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv()

class Settings:
    MAIN_BASE_URL: str = os.getenv("BEFS_MAIN_BASE_URL") if os.getenv("BEFS_MAIN_BASE_URL") else "db/store.sqlite3"
    FASTAPI_SERVER_HOST: str = os.getenv("BEFS_FASTAPI_SERVER_HOST") if os.getenv("BEFS_FASTAPI_SERVER_HOST") else "0.0.0.0"
    FASTAPI_SERVER_PORT: int = os.getenv("BEFS_FASTAPI_SERVER_PORT") if os.getenv("BEFS_FASTAPI_SERVER_PORT") else 5000
    SECRET_KEY: str = os.getenv("BEFS_SECRET_KEY") if os.getenv("BEFS_SECRET_KEY") else secrets.token_hex(12)
    API_KEY: str = os.getenv("BEFS_API_KEY") if os.getenv("BEFS_API_KEY") else secrets.token_hex(12)


settings = Settings()
