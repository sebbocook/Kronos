"""
Kronos als Web-API für InstaPods.

Stellt einen POST-Endpunkt /forecast bereit, der historische OHLC(V)-Daten
entgegennimmt und eine Kronos-Vorhersage MIT Unsicherheitsbändern zurückgibt.

- Modell wird beim Start des Servers EINMAL geladen (bleibt danach im RAM).
- Statt eines einzelnen gemittelten Pfades werden NUM_PATHS unabhängige
  Forecast-Läufe durchgeführt; pro future_timestamp werden Median sowie
  10./90. Perzentil für open/high/low/close (und Median für volume)
  zurückgegeben. Das macht die Unsicherheit über den Horizont hinweg
  sichtbar - wichtig bei längeren Swing-Trade-Horizonten (z. B. 4 Wochen),
  wo ein einzelner Punktwert am Ende der Prognose kaum belastbar wäre.
- load_dotenv() liest die .env-Datei im Pod explizit ein.
- Optionaler Schutz per API-Key über den Header "X-API-Key".
"""

import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

load_dotenv()

from model import Kronos, KronosTokenizer, KronosPredictor

MODEL_NAME = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MAX_CONTEXT = int(os.environ.get("KRONOS_MAX_CONTEXT", "512"))
API_KEY = os.environ.get("KRONOS_API_KEY")
DEFAULT_NUM_PATHS = int(os.environ.get("KRONOS_NUM_PATHS", "25"))

state = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)
    state["predictor"] = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)
    print(f"Kronos geladen: {MODEL_NAME} / {TOKENIZER_NAME}")
    print(f"API-Key-Schutz aktiv: {API_KEY is not None}")
    print(f"Standard-Pfadanzahl für Perzentil-Bänder: {DEFAULT_NUM_PATHS}")
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
    num_paths: int | None = Field(default=None, description="Anzahl unabhängiger Forecast-Läufe für die Perzentil-Bänder. Default: KRONOS_NUM_PATHS env var.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["predictor"] is not None,
        "model": MODEL_NAME,
        "api_key_protected": API_KEY is not None,
        "default_num_paths": DEFAULT_NUM_PATHS,
    }


@app.post("/forecast")
def forecast(req: ForecastRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    if state["predictor"] is None:
        raise HTTPException(503, "Modell ist noch nicht bereit, bitte kurz warten")

    if len(req.history) < 2:
        raise HTTPException(400, "Mindestens 2 historische Kerzen nötig")

    num_paths = req.num_paths or DEFAULT_NUM_PATHS
    pred_len = len(req.future_timestamps)

    df = pd.DataFrame([c.dict() for c in req.history])
    x_timestamp = pd.to_datetime(df["timestamp"])
    y_timestamp = pd.to_datetime(pd.Series(req.future_timestamps))
    df = df.drop(columns=["timestamp"])
    has_volume = not df["volume"].isna().all()
    if not has_volume:
        df = df.drop(columns=["volume"])

    try:
        paths = []
        for _ in range(num_paths):
            pred_df = state["predictor"].predict(
                df=df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=req.temperature,
                sample_count=1,
            )
            paths.append(pred_df)
    except Exception as e:
        raise HTTPException(500, f"Vorhersage fehlgeschlagen: {e}")

    forecast_out = []
    for i, ts in enumerate(req.future_timestamps):
        opens = np.array([p.iloc[i]["open"] for p in paths])
        highs = np.array([p.iloc[i]["high"] for p in paths])
        lows = np.array([p.iloc[i]["low"] for p in paths])
        closes = np.array([p.iloc[i]["close"] for p in paths])

        entry = {
            "timestamp": ts,
            "open_median": float(np.median(opens)),
            "high_median": float(np.median(highs)),
            "high_p90": float(np.percentile(highs, 90)),
            "low_median": float(np.median(lows)),
            "low_p10": float(np.percentile(lows, 10)),
            "close_median": float(np.median(closes)),
            "close_p10": float(np.percentile(closes, 10)),
            "close_p90": float(np.percentile(closes, 90)),
        }
        if has_volume and "volume" in paths[0].columns:
            volumes = np.array([p.iloc[i]["volume"] for p in paths])
            entry["volume_median"] = float(np.median(volumes))

        forecast_out.append(entry)

    return {"forecast": forecast_out, "paths_computed": num_paths}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
