#!/usr/bin/env python3
"""
Entrena y evalúa un MLP multiclase en TensorFlow/Keras usando validación cruzada 5-fold,
incorpora EarlyStopping y ReduceLROnPlateau en cada fold, visualiza y suaviza curvas de pérdida
y accuracy, aplica regularización y ruido, y guarda métricas e imágenes de los reportes.
"""
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Sequential, regularizers, callbacks
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import matplotlib.pyplot as plt

# Parámetros
OPT_PARAMS = {
    'neurons': 64,
    'activation': 'relu',
    'optimizer': 'rmsprop',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'l2': 1e-4,
    'dropout': 0.7,
    'noise_std': 0.1,
    'smooth_factor': 0.8,
    'n_splits': 5,
    'es_patience': 10,
    'lr_factor': 0.5,
    'lr_patience': 5
}

# Directorios
dir_list = ['models_tf', 'reports']
for d in dir_list:
    os.makedirs(d, exist_ok=True)

# 1) Carga y preprocesa datos
df = pd.read_csv('../buenos/dataset_ransomware_benign_augmented.csv')
df.drop(columns=['ID'], inplace=True)
df['Family'].fillna('None', inplace=True)
# Filtrar infectados y preparar multiclase
df_multi = df[df['infected'] == 1]
X = df_multi.drop(columns=['infected','Family']).values
y_labels = df_multi['Family'].values
# Codificar etiquetas
y_enc = LabelEncoder().fit(y_labels)
y = y_enc.transform(y_labels)

# Escalado global antes de CV
scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)
joblib.dump(scaler, os.path.join('models_tf', 'scaler_multiclass.joblib'))

# Funciones de construcción de modelo y suavizado
def build_model(input_dim):
    reg = regularizers.l2(OPT_PARAMS['l2'])
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.GaussianNoise(OPT_PARAMS['noise_std']),
        layers.Dense(OPT_PARAMS['neurons'], activation=OPT_PARAMS['activation'], kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(OPT_PARAMS['dropout']),
        layers.Dense(OPT_PARAMS['neurons']//2, activation=OPT_PARAMS['activation'], kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Dropout(OPT_PARAMS['dropout']),
        layers.Dense(len(y_enc.classes_), activation='softmax')
    ])
    opt = tf.keras.optimizers.get(OPT_PARAMS['optimizer'])
    opt.learning_rate = OPT_PARAMS['learning_rate']
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def smooth_curve(values, factor):
    smooth = []
    for v in values:
        smooth.append(v if not smooth else smooth[-1]*factor + v*(1-factor))
    return smooth

# Acumular curvas y métricas
all_hist = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
metrics = []

# 2) Validación cruzada 5-fold con callbacks
kf = KFold(n_splits=OPT_PARAMS['n_splits'], shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(kf.split(X_s), 1):
    print(f"Fold {i}/{OPT_PARAMS['n_splits']}")
    X_train, X_val = X_s[train_idx], X_s[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model(X_train.shape[1])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=OPT_PARAMS['es_patience'], restore_best_weights=True)
    rlrp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=OPT_PARAMS['lr_factor'], patience=OPT_PARAMS['lr_patience'], verbose=1)

    hist = model.fit(
        X_train, y_train,
        epochs=OPT_PARAMS['epochs'], batch_size=OPT_PARAMS['batch_size'], verbose=0,
        validation_data=(X_val, y_val), callbacks=[es, rlrp]
    )

    # Acumular curvas
    for k in all_hist:
        all_hist[k].append(hist.history.get(k, []))

    # Evaluar fold
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_proba = model.predict(X_val)
    metrics.append({
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
        'AUC': roc_auc_score(tf.keras.utils.to_categorical(y_val), y_proba, multi_class='ovr')
    })

# 3) Guardar métricas promedio CV
df_metrics = pd.DataFrame(metrics).mean().to_frame().T
df_metrics.to_csv('models_tf/multiclass_mlp_kfold_metrics.csv', index=False)
print("Métricas CV guardadas en models_tf/multiclass_mlp_kfold_metrics.csv")

# 4) Promediar y suavizar curvas
min_len = min(len(h) for h in all_hist['loss'])
avg_hist = {}
for k in all_hist:
    arr = np.array([h[:min_len] for h in all_hist[k]])
    mean = arr.mean(axis=0)
    avg_hist[k] = smooth_curve(mean, OPT_PARAMS['smooth_factor'])

epochs = range(1, min_len+1)

# 5) Graficar
def plot_curve(x, y1, y2, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y1, label=f'{ylabel} (train)')
    plt.plot(x, y2, label=f'{ylabel} (val)')
    plt.title(title)
    plt.xlabel('Época')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('reports', filename))
    plt.close()

plot_curve(epochs, avg_hist['loss'], avg_hist['val_loss'], 'Loss', 'Curva de Pérdida CV', 'loss_kfold.png')
plot_curve(epochs, avg_hist['accuracy'], avg_hist['val_accuracy'], 'Accuracy', 'Curva de Accuracy CV', 'accuracy_kfold.png')

print("Gráficas CV guardadas en carpeta 'reports'")
