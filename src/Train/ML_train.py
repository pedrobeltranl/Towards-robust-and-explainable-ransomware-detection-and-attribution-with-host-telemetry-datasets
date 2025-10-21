import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt

# Crear directorio de modelos
os.makedirs('./models', exist_ok=True)

# Cargar dataset
df = pd.read_csv('../buenos/dataset_ransomware_benign_augmented.csv')
df = df.drop(columns=['ID'])
df['Family'] = df['Family'].fillna('None')

# Características y objetivos
X = df.drop(columns=['infected', 'Family'])
y_bin = df['infected']
mask = y_bin == 1
X_multi = X.loc[mask]
y_multi = df.loc[mask, 'Family']

# Dividir conjuntos
test_size = 0.2
rnd = 42
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X, y_bin, test_size=test_size, random_state=rnd, stratify=y_bin
)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_multi, y_multi, test_size=test_size, random_state=rnd, stratify=y_multi
)

# Escalar
def scale_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

Xb_train_s, Xb_test_s = scale_data(Xb_train, Xb_test)
Xm_train_s, Xm_test_s = scale_data(Xm_train, Xm_test)

# Guardar datos de test
np.savez(
    './models/test_data.npz',
    Xb_test=Xb_test_s,
    yb_test=yb_test,
    Xm_test=Xm_test_s,
    ym_test=ym_test
)
print('✅ Test data saved to ./models/test_data.npz')

# Definir modelos
def get_models():
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=rnd),
        'RandomForest': RandomForestClassifier(random_state=rnd),
        'SVC': SVC(probability=True, random_state=rnd),
        'GradientBoosting': GradientBoostingClassifier(random_state=rnd),
        'KNeighbors': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(random_state=rnd)
    }

# Parámetros simplificados para multiclas
params_multi = {
    'LogisticRegression': {'C': [1]},
    'RandomForest': {'n_estimators': [100]},
    'SVC': {'kernel': ['rbf']},
    'GradientBoosting': {'n_estimators': [100]},
    'KNeighbors': {'n_neighbors': [5]},
    'AdaBoost': {'n_estimators': [50]}
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rnd)

# Diccionarios/listas para guardar resultados
summary_bin, summary_multi = [], []
roc_data, cm_bin, cm_multi = {}, {}, {}

# Loop para binario y multiclase
for stage, (X_tr, X_te, y_tr, y_te, cm_dict, summary_list) in {
    'Binary': (Xb_train_s, Xb_test_s, yb_train, yb_test, cm_bin, summary_bin),
    'Multiclass': (Xm_train_s, Xm_test_s, ym_train, ym_test, cm_multi, summary_multi)
}.items():
    for name, model in get_models().items():
        # Grid search
        params = params_multi.get(name, {}) if stage == 'Multiclass' else {}
        grid = GridSearchCV(model, params, cv=cv, n_jobs=-1)
        grid.fit(X_tr, y_tr)
        best = grid.best_estimator_
        joblib.dump(best, f"./models/{name}_{stage}.joblib")

        # Predicción
        y_pred = best.predict(X_te)

        # ===== Métricas "normales" y "balanceadas" =====
        # Accuracy y Balanced Accuracy
        acc = accuracy_score(y_te, y_pred)
        bal_acc = balanced_accuracy_score(y_te, y_pred)

        # Precision/Recall/F1: weighted y macro (macro = "balanceado por clases")
        prec_w = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        rec_w  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
        f1_w   = f1_score(y_te, y_pred, average='weighted', zero_division=0)

        prec_macro = precision_score(y_te, y_pred, average='macro', zero_division=0)
        rec_macro  = recall_score(y_te, y_pred, average='macro', zero_division=0)
        f1_macro   = f1_score(y_te, y_pred, average='macro', zero_division=0)

        # F1 binario explícito cuando procede (útil para comparar con weighted/macro)
        f1_binary = None
        if stage == 'Binary':
            # Asumimos etiqueta positiva = 1 (ajústalo si tu dataset usa otra)
            f1_binary = f1_score(y_te, y_pred, average='binary', zero_division=0, pos_label=1)

        # Guardar al resumen
        row = {
            'Stage': stage, 'Model': name,
            'Accuracy': acc, 'Balanced_Accuracy': bal_acc,
            'Precision_weighted': prec_w, 'Precision_macro': prec_macro,
            'Recall_weighted': rec_w, 'Recall_macro': rec_macro,
            'F1_weighted': f1_w, 'F1_macro': f1_macro
        }
        if f1_binary is not None:
            row['F1_binary'] = f1_binary

        summary_list.append(row)

        # Matriz de confusión normalizada (por filas)
        cm = confusion_matrix(y_te, y_pred, normalize='true')
        cm_dict[name] = cm

        # ROC sólo para binario
        if stage == 'Binary':
            if hasattr(best, "predict_proba"):
                y_score = best.predict_proba(X_te)[:, 1]
            elif hasattr(best, "decision_function"):
                # Escalar decision_function a [0,1] para construir una pseudo-proba
                dec = best.decision_function(X_te)
                dec = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
                y_score = dec
            else:
                # Si no hay score continuo, no se puede calcular una curva ROC significativa
                y_score = None

            if y_score is not None:
                fpr, tpr, _ = roc_curve(y_te, y_score)
                roc_data[name] = (fpr, tpr, auc(fpr, tpr))

# ======= Exportar métricas a CSV =======
df_bin = pd.DataFrame(summary_bin)
df_multi = pd.DataFrame(summary_multi)

# Orden de columnas (binario y multiclase)
cols_bin = [
    'Stage', 'Model', 'Accuracy', 'Balanced_Accuracy',
    'Precision_weighted', 'Precision_macro',
    'Recall_weighted', 'Recall_macro',
    'F1_binary', 'F1_weighted', 'F1_macro'
]
cols_multi = [
    'Stage', 'Model', 'Accuracy', 'Balanced_Accuracy',
    'Precision_weighted', 'Precision_macro',
    'Recall_weighted', 'Recall_macro',
    'F1_weighted', 'F1_macro'
]

# Asegurar columnas presentes (por si algún modelo no produjo algo)
for c in cols_bin:
    if c not in df_bin.columns:
        df_bin[c] = np.nan
for c in cols_multi:
    if c not in df_multi.columns:
        df_multi[c] = np.nan

df_bin = df_bin[cols_bin]
df_multi = df_multi[cols_multi]

# Guardar
bin_path = './models/metrics_binary.csv'
multi_path = './models/metrics_multiclass.csv'
all_path = './models/metrics_all.csv'

df_bin.to_csv(bin_path, index=False)
df_multi.to_csv(multi_path, index=False)
pd.concat([df_bin, df_multi], ignore_index=True).to_csv(all_path, index=False)

print('📄 CSVs de métricas guardados:')
print(f' - {bin_path}')
print(f' - {multi_path}')
print(f' - {all_path}')

# --- Radar chart MULTICLASS (incluye métricas balanceadas) ---
# Para que el radar no se vuelva ilegible, usamos un conjunto compacto y balanceado:
# Accuracy, Balanced_Accuracy, Precision_macro, Recall_macro, F1_macro
labels = ['Accuracy', 'Balanced_Accuracy', 'Precision_macro', 'Recall_macro', 'F1_macro']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for s in summary_multi:
    values = [s[m] for m in labels]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=s['Model'])
    ax.fill(angles, values, alpha=0.1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title('Radar Chart - Multiclass Models (balanced & macro metrics)', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('./models/radar_multiclass.png')
plt.close()

# --- Heatmap multiclas ---
n = len(cm_multi)
cols = 3
rows = int(np.ceil(n / cols)) if n > 0 else 1
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = np.atleast_1d(axes).flatten()
for idx, (name, cm) in enumerate(cm_multi.items()):
    ax = axes[idx]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im, ax=ax)
for j in range(idx + 1, rows * cols):
    fig.delaxes(axes[j])
plt.suptitle('Normalized Confusion Matrices - Multiclass')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./models/confusion_heatmaps_multiclass.png')
plt.close()

# --- (Opcional) Curvas ROC binario por modelo ---
if len(roc_data) > 0:
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Binary Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./models/roc_binary.png')
    plt.close()

print('✅ Gráficos guardados en ./models (radar_multiclass.png, confusion_heatmaps_multiclass.png, roc_binary.png si aplica)')
