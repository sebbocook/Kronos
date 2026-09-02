"""
Kronos als Web-API für InstaPods.

Stellt einen POST-Endpunkt /forecast bereit, der historische OHLC(V)-Daten
entgegennimmt und eine Kronos-Vorhersage MIT Unsicherheitsbändern zurückgibt.

Performance-Optimierungen (alle per Umgebungsvariable an/abschaltbar,
damit man bei Problemen einfach zurückschalten kann, ohne Code zu ändern):
- KRONOS_CPU_THREADS: explizite Thread-Anzahl für PyTorch (Default: 2,
  passend zum InstaPods Build-Plan). Vermeidet, dass PyTorch im Container
  eine falsche Kernanzahl errät.
- KRONOS_USE_QUANTIZATION (Default: true): dynamische int8-Quantisierung
  der Linear-Layer. Meist 20-40% schnellere Inferenz auf CPU, minimaler
  Genauigkeitsverlust bei einem ohnehin kleinen Modell. Gilt als etablierte,
  risikoarme Optimierung.
- KRONOS_USE_COMPILE (Default: false): torch.compile() beim Start.
  Kann zusätzlich beschleunigen, ist aber weniger vorhersagbar (kann bei
  manchen Modellarchitekturen scheitern oder wirkungslos bleiben) - daher
  standardmäßig aus, testweise aktivierbar.
Beide Optimierungen sind mit try/except abgesichert: schlägt eine fehl,
läuft der Server einfach mit dem unveränderten Modell weiter (Log-Ausgabe
zeigt das an), statt abzustürzen.
"""

import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
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

CPU_THREADS = int(os.environ.get("KRONOS_CPU_THREADS", "2"))
USE_QUANTIZATION = os.environ.get("KRONOS_USE_QUANTIZATION", "true").lower() == "true"
USE_COMPILE = os.environ.get("KRONOS_USE_COMPILE", "false").lower() == "true"

torch.set_num_threads(CPU_THREADS)

state = {"predictor": None, "quantization_active": False, "compile_active": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(MODEL_NAME)

    if USE_QUANTIZATION:
        try:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            state["quantization_active"] = True
            print("Dynamische Quantisierung aktiviert")
        except Exception as e:
            print(f"Quantisierung fehlgeschlagen, verwende Original-Modell: {e}")

    if USE_COMPILE:
        try:
            model = torch.compile(model)
            state["compile_active"] = True
            print("torch.compile aktiviert")
        except Exception as e:
            print(f"torch.compile fehlgeschlagen, verwende unkompiliertes Modell: {e}")

    state["predictor"] = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)
    print(f"Kronos geladen: {MODEL_NAME} / {TOKENIZER_NAME}")
    print(f"API-Key-Schutz aktiv: {API_KEY is not None}")
    print(f"CPU-Threads: {CPU_THREADS}, Quantisierung: {state['quantization_active']}, Compile: {state['compile_active']}")
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
        "cpu_threads": CPU_THREADS,
        "quantization_active": state["quantization_active"],
        "compile_active": state["compile_active"],
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
