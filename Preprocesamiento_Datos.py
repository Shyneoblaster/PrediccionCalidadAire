"""
Preprocesamiento_Datos.py — Carga, limpieza y preparación de secuencias para LSTM.
Dataset: Beijing PM2.5 (UCI ML Repository)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
WIND_MAP = {"NW": 0, "NE": 1, "SE": 2, "cv": 3}
SEQ_LEN  = 24   # horas de historia
HORIZON  = 6    # horas a predecir


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cbwd"] = df["cbwd"].map(WIND_MAP).fillna(0).astype(int)
    df["pm2.5"] = df["pm2.5"].interpolate(method="linear").bfill()
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    return df[FEATURES]


def make_sequences(data: np.ndarray, seq_len: int = SEQ_LEN,
                   horizon: int = HORIZON):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len: i + seq_len + horizon, 0])  # solo pm2.5
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare(path: str):
    df = load_and_clean(path)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    split = int(len(scaled) * 0.8)
    X_tr, y_tr = make_sequences(scaled[:split])
    X_te, y_te = make_sequences(scaled[split:])
    return X_tr, y_tr, X_te, y_te, scaler, df
