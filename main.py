import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Liste des origines autorisées (vos domaines Cloudflare Pages)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
allowed_origins = [o.strip() for o in allowed_origins if o.strip()]

# En dev, autoriser aussi localhost
if not allowed_origins or "*" in allowed_origins:
    allowed_origins = ["*"]

app = FastAPI(title="NVIDIA NIM Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "NIM Proxy"}

@app.get("/api/status")
async def status():
    """Vérifie que la clé API fonctionne."""
    if not NVIDIA_API_KEY:
        return {"online": False, "error": "Clé API manquante"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{NIM_BASE_URL}/models",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"}
            )
            return {"online": r.status_code == 200}
    except Exception as e:
        return {"online": False, "error": str(e)}

@app.post("/api/nim")
async def proxy_nim(request: Request):
    """Proxy vers l'API chat/completions de NVIDIA NIM."""
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API NVIDIA non configurée")
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON invalide")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{NIM_BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {NVIDIA_API_KEY}"
                },
                json=body
            )
            
            data = response.json()
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=response.status_code,
                    content=data
                )
            
            return data
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout NVIDIA NIM")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))