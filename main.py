"""
main.py — Pipeline principal + 3 casos de prueba.
Uso: python main.py --data <ruta_csv>
"""
import argparse
import numpy as np
import tensorflow as tf

from preprocessing import prepare, FEATURES, SEQ_LEN, HORIZON
from lstm_model    import build_lstm, train, evaluate
from expert_system import razonar


# ── Entrenamiento ────────────────────────────────────────────────────────────
def run_training(data_path: str):
    print("\n[1/3] Preprocesando datos...")
    X_tr, y_tr, X_te, y_te, scaler, df = prepare(data_path)
    print(f"  Train: {X_tr.shape}  |  Test: {X_te.shape}")

    print("\n[2/3] Entrenando modelo LSTM...")
    model = build_lstm(SEQ_LEN, len(FEATURES), HORIZON)
    model.summary()
    train(model, X_tr, y_tr, X_te, y_te, epochs=30, batch_size=256)

    print("\n[3/3] Evaluando modelo...")
    metrics = evaluate(model, X_te, y_te, scaler)
    print(f"  MSE : {metrics['MSE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²  : {metrics['R2']:.4f}")

    model.save("lstm_pm25.keras")
    print("  Modelo guardado → lstm_pm25.keras")
    return model, scaler, df


# ── Predicción + SE ──────────────────────────────────────────────────────────
def predict_and_reason(model, scaler, window: np.ndarray) -> dict:
    """Predice 6 horas de PM2.5 y activa el motor de inferencia."""
    n_feat = scaler.n_features_in_
    scaled_win = scaler.transform(window)
    X = scaled_win[np.newaxis, :, :]                        # (1, SEQ, FEAT)
    pred_scaled = model.predict(X, verbose=0)[0]            # (6,)

    # Desescalar las 6 predicciones
    dummy = np.zeros((HORIZON, n_feat))
    dummy[:, 0] = pred_scaled
    pred_real = scaler.inverse_transform(dummy)[:, 0].tolist()

    return razonar(pred_real)


# ── Mostrar resultado completo ───────────────────────────────────────────────
def mostrar_resultado(nombre: str, r: dict):
    print(f"\n{'─'*60}")
    print(f"  📍 {nombre}")
    print(f"{'─'*60}")
    print(f"  Predicción PM2.5 próximas 6 horas:")
    for linea in r["horas_detalle"]:
        print(linea)
    print(f"\n  Resumen del peor escenario:")
    print(f"  {r['cadena_razonamiento']}")
    print(f"\n  Color NOM-172  : {r['color']}")
    print(f"  Rec. general   : {r['rec_general']}")
    print(f"  Rec. sensibles : {r['rec_sensibles']}")


# ── Casos de prueba ──────────────────────────────────────────────────────────
def build_test_window(pm25, temp, pres, dewp=-5.0, iws=5.0, is_=0, ir=0):
    # FEATURES = ["pm2.5","DEWP","TEMP","PRES","Iws","Is","Ir"]
    row = [pm25, dewp, temp, pres, iws, is_, ir]
    return np.tile(row, (SEQ_LEN, 1)).astype(float)


CASOS = {
    "Día limpio":              build_test_window(15,  15, 1015),
    "Contaminación moderada":  build_test_window(60,  10, 1010),
    "Contingencia (muy alto)": build_test_window(160,  5, 1005, iws=1.0),
}


def demo_cases(model, scaler):
    print("\n" + "=" * 60)
    print("  DEMOSTRACIÓN — 3 CASOS DE PRUEBA")
    print("=" * 60)
    for nombre, ventana in CASOS.items():
        r = predict_and_reason(model, scaler, ventana)
        mostrar_resultado(nombre, r)
    print("=" * 60)


# ── Entrada interactiva ──────────────────────────────────────────────────────
def interactive_mode(model, scaler):
    print("\n[MODO INTERACTIVO] Ingresa condiciones meteorológicas actuales.")
    try:
        pm25 = float(input("  PM2.5 actual (µg/m³): "))
        temp = float(input("  Temperatura (°C)    : "))
        pres = float(input("  Presión (hPa)       : "))
        dewp = float(input("  Punto de rocío (°C) : "))
        iws  = float(input("  Vel. viento (m/s)   : "))
    except (ValueError, EOFError):
        print("  Entrada inválida, omitiendo modo interactivo.")
        return

    ventana = build_test_window(pm25, temp, pres, dewp, iws)
    r = predict_and_reason(model, scaler, ventana)
    mostrar_resultado("Condiciones ingresadas", r)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema Híbrido LSTM+SE — Calidad del Aire")
    parser.add_argument("--data",  default="PRSA_data_2010.1.1-2014.12.31.csv")
    parser.add_argument("--model", default=None, help="Ruta a modelo .keras ya entrenado")
    parser.add_argument("--no-interactive", action="store_true")
    args = parser.parse_args()

    if args.model:
        print(f"Cargando modelo desde {args.model}...")
        _, _, _, _, scaler, df = prepare(args.data)
        model = tf.keras.models.load_model(args.model)
    else:
        model, scaler, df = run_training(args.data)

    demo_cases(model, scaler)

    if not args.no_interactive:
        interactive_mode(model, scaler)