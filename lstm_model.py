"""
lstm_model.py — LSTM con TensorFlow/Keras.
Métricas: MSE, RMSE, R²
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score


def build_lstm(seq_len: int, n_features: int, horizon: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(horizon),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train(model: tf.keras.Model, X_tr, y_tr, X_te, y_te,
          epochs: int = 30, batch_size: int = 256):
    cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=epochs, batch_size=batch_size,
        callbacks=[cb], verbose=1,
    )
    return history


def evaluate(model: tf.keras.Model, X_te, y_te, scaler) -> dict:
    y_pred_s = model.predict(X_te, verbose=0)

    def descale(arr):
        dummy = np.zeros((arr.shape[0] * arr.shape[1], scaler.n_features_in_))
        dummy[:, 0] = arr.flatten()
        return scaler.inverse_transform(dummy)[:, 0].reshape(arr.shape)

    y_true = descale(y_te)
    y_pred = descale(y_pred_s)

    mse  = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true.flatten(), y_pred.flatten())
    return {"MSE": mse, "RMSE": rmse, "R2": r2,
            "y_true": y_true, "y_pred": y_pred}