"""
Kronos als Web-API für InstaPods.

Stellt einen POST-Endpunkt /forecast bereit, der historische OHLC(V)-Daten
entgegennimmt und eine Kronos-Vorhersage als JSON zurückgibt.

Modell: NeoQuasar/Kronos-small (24.7M Parameter) + Kronos-Tokenizer-base.
Läuft auf CPU -> passt in 1-2GB RAM (Build-Plan). Für sehr knappe Pods
kann auf "NeoQuasar/Kronos-mini" umgestellt werden (siehe unten).
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# "model" ist der Ordner aus dem Kronos-Repo (model/ liegt neben dieser Datei)
from model import Kronos, KronosTokenizer, KronosPredictor

app = FastAPI(title="Kronos Forecast API")

MODEL_NAME = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MAX_CONTEXT = int(os.environ.get("KRONOS_MAX_CONTEXT", "512"))

predictor = None  # wird beim ersten Request geladen (spart RAM beim Kaltstart)


def get_predictor() -> KronosPredictor:
    global predictor
    if predictor is None:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(MODEL_NAME)
        predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)
    return predictor


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
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if len(req.history) < 2:
        raise HTTPException(400, "Mindestens 2 historische Kerzen nötig")

    df = pd.DataFrame([c.dict() for c in req.history])
    x_timestamp = pd.to_datetime(df["timestamp"])
    y_timestamp = pd.to_datetime(pd.Series(req.future_timestamps))
    df = df.drop(columns=["timestamp"])
    if df["volume"].isna().all():
        df = df.drop(columns=["volume"])

    try:
        pred_df = get_predictor().predict(
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
