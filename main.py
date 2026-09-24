import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
allowed_origins = [o.strip() for o in allowed_origins if o.strip()]

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

# ============================================================
# SYSTEM PROMPT : force l'IA à produire un site HTML complet
# ============================================================
SYSTEM_PROMPT = """Tu es un développeur web senior expert en HTML/CSS/JavaScript et en design UI/UX.

Ta mission : générer un site web COMPLET et UNIQUE en te basant STRICTEMENT sur le prompt de l'utilisateur.

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT avec un objet JSON valide (sans markdown, sans ```, sans texte avant/après).
2. Le JSON doit contenir EXACTEMENT ces clés :
{
  "name": "nom court du site",
  "tagline": "slogan court et percutant",
  "theme": {
    "primary": "#hex",
    "secondary": "#hex",
    "background": "#hex",
    "text": "#hex",
    "font": "nom de police Google Fonts (ex: 'Inter', 'Poppins', 'Playfair Display')"
  },
  "html": "code HTML COMPLET et AUTONOME du site (avec <style> intégré, pas de dépendances externes sauf Google Fonts)"
}

RÈGLES DE GÉNÉRATION DU HTML :
- Le HTML doit être un document complet : <!DOCTYPE html><html>...<head>...<body>...</body></html>
- Le CSS doit être DANS une balise <style> dans le <head>.
- Utilise UNIQUEMENT Google Fonts en externe (via <link>), RIEN d'autre d'externe.
- Le design doit être MODERNE, professionnel et RESPECTER le style demandé dans le prompt.
- Crée une structure adaptée au type de site demandé :
  * portfolio → hero + projets + à propos + contact
  * business/landing → hero + services + avantages + témoignages + CTA + contact
  * blog → header + articles + sidebar + footer
  * ecommerce → header + produits + panier + footer
  * restaurant → hero + menu + galerie + réservation + contact
- INVENTE du contenu RICHE et COHÉRENT avec le prompt (titres, descriptions, noms, chiffres, etc.).
- Le site doit être RESPONSIVE (media queries).
- Utilise les couleurs du thème définies dans "theme".
- Ajoute des animations subtiles (hover, transitions).
- NE PAS inclure d'images externes (utilise des gradients, SVG inline, ou emojis).
- Le HTML doit être PRÊT À AFFICHER tel quel dans un iframe (taille min 1200px de large).

IMPORTANT : chaque génération doit être UNIQUE. Varie les layouts, les couleurs, les structures selon le prompt."""


@app.get("/")
async def root():
    return {"status": "ok", "service": "NIM Proxy"}


@app.get("/api/status")
async def status():
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
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API NVIDIA non configurée")
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON invalide")
    
    # Injecter le system prompt renforcé s'il n'y en a pas
    if "messages" in body:
        has_system = any(m.get("role") == "system" for m in body["messages"])
        if not has_system:
            body["messages"].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    # Augmenter max_tokens pour laisser la place au HTML complet
    if "max_tokens" not in body or body["max_tokens"] < 8192:
        body["max_tokens"] = 8192
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
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
        raise HTTPException(status_code=504, detail="Timeout NVIDIA NIM (le modèle est lent)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))