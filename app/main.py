import os
import secrets
import threading

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .ocr import EQUIPAMENTOS, Reader, UnknownDevice

API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

app = FastAPI(title="OCR Analab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

reader = Reader()
# a sessão ONNX já usa todos os núcleos; uma leitura por vez por processo
lock = threading.Lock()


def check_key(x_api_key: str | None) -> None:
    if not API_KEYS:
        return
    if not x_api_key or not any(secrets.compare_digest(x_api_key, k) for k in API_KEYS):
        raise HTTPException(status_code=401, detail="API key inválida")


def run(raw: bytes, equipamento: str | None) -> dict:
    with lock:
        return reader.read(raw, equipamento)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/equipamentos")
async def equipamentos():
    return [{"id": k, "nome": v} for k, v in EQUIPAMENTOS.items()]


@app.post("/api/ocr")
async def ocr(
    file: UploadFile = File(...),
    equipamento: str | None = Form(default=None, description="hematoclin-3.0, abx-micros-es60 ou auto"),
    equipamento_q: str | None = Query(default=None, alias="equipamento"),
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Arquivo vazio")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Arquivo maior que {MAX_UPLOAD_MB} MB")
    try:
        return await run_in_threadpool(run, raw, equipamento or equipamento_q)
    except UnknownDevice as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
