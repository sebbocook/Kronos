"""
Kronos als Web-API für InstaPods.

Stellt einen POST-Endpunkt /forecast bereit, der historische OHLC(V)-Daten
entgegennimmt und eine Kronos-Vorhersage als JSON zurückgibt.

- Modell wird beim Start des Servers EINMAL geladen (bleibt danach im RAM),
  damit jeder API-Request sofort beantwortet wird.
- Optionaler Schutz per API-Key: Wenn die Umgebungsvariable KRONOS_API_KEY
  auf dem Pod gesetzt ist, muss jeder Request den Header
  "X-API-Key: <derselbe Wert>" mitschicken. Ist die Variable nicht gesetzt,
  ist der Endpunkt offen (nur für schnelle Tests empfohlen).
"""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# "model" ist der Ordner aus dem Kronos-Repo (model/ liegt neben dieser Datei)
from model import Kronos, KronosTokenizer, KronosPredictor

MODEL_NAME = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MAX_CONTEXT = int(os.environ.get("KRONOS_MAX_CONTEXT", "512"))
API_KEY = os.environ.get("KRONOS_API_KEY")  # None = kein Schutz aktiv

state = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Läuft einmalig beim Start des Servers -> Modell liegt danach dauerhaft im RAM
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    state["predictor"] = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)
    print(f"Kronos geladen: {MODEL_NAME} / {TOKENIZER_NAME}")
    yield
    state["predictor"] = None


app = FastAPI(title="Kronos Forecast API", lifespan=lifespan)


def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Ungültiger oder fehlender X-API-Key Header")


class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class ForecastRequest(BaseModel):
    history: list[Candle] = Field(..., description="Historische K-Line-Daten, älteste zuerst")
    future_timestamps: list[str] = Field(..., description="Zeitstempel, für die vorhergesagt werden soll")
    temperature: float = 1.0
    sample_count: int = 1


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["predictor"] is not None, "model": MODEL_NAME}


@app.post("/forecast")
def forecast(req: ForecastRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    if state["predictor"] is None:
        raise HTTPException(503, "Modell ist noch nicht bereit, bitte kurz warten")

    if len(req.history) < 2:
        raise HTTPException(400, "Mindestens 2 historische Kerzen nötig")

    df = pd.DataFrame([c.dict() for c in req.history])
    x_timestamp = pd.to_datetime(df["timestamp"])
    y_timestamp = pd.to_datetime(pd.Series(req.future_timestamps))
    df = df.drop(columns=["timestamp"])
    if df["volume"].isna().all():
        df = df.drop(columns=["volume"])

    try:
        pred_df = state["predictor"].predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=len(req.future_timestamps),
            T=req.temperature,
            sample_count=req.sample_count,
        )
    except Exception as e:
        raise HTTPException(500, f"Vorhersage fehlgeschlagen: {e}")

    return {"forecast": pred_df.reset_index().to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
