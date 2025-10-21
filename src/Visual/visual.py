#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PCA + UMAP con:
- Filtro de filas sin 'Family' cuando se grafican vistas por 'Family'
- UMAP no supervisado entrenado en el subconjunto mostrado
- Eliminación de outliers en 'Family' del UMAP no supervisado:
  * Regla explícita: remover puntos con UMAP-1 > UMAP1_MAX (default 15.0)
- UMAP supervisado con y codificada a enteros
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silencia TF/CUDA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ======== Parámetro de corte para outliers en UMAP-1 (Family) ========
UMAP1_MAX = 15.0  # elimina puntos con UMAP-1 > 15

# UMAP
try:
    import umap.umap_ as umap
except Exception as e:
    raise SystemExit(
        "No se encontró 'umap-learn'. Instala con: pip install umap-learn\n"
        f"Detalle: {e}"
    )

# ------------------------------
# Config y salidas
# ------------------------------
base_dir = 'imagenes'
for pc in ['pc1', 'pc2']:
    for label in ['infected', 'Family']:
        os.makedirs(f'{base_dir}/{pc}/{label}', exist_ok=True)
for mode in ['umap_unsupervised', 'umap_supervised']:
    for label in ['infected', 'Family']:
        os.makedirs(f'{base_dir}/{mode}/{label}', exist_ok=True)

# ------------------------------
# Datos
# ------------------------------
csv_path = '../buenos/dataset_ransomware_benign_augmented.csv'
df = pd.read_csv(csv_path, low_memory=False)

# ------------------------------
# Numéricas, escalado y PCA global
# ------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in ['ID', 'infected']:
    if col in numeric_cols:
        numeric_cols.remove(col)

X_full = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
X_full = X_full.fillna(X_full.mean(numeric_only=True))

scaler = StandardScaler().fit(X_full)
X_scaled_full = scaler.transform(X_full)

pca = PCA().fit(X_scaled_full)
X_pca_full = pca.transform(X_scaled_full)

# ------------------------------
# Función
# ------------------------------
def generate_plots(pc_index: int, label_col: str):
    pc_name = f'pc{pc_index+1}'
    folder_pca = f'{base_dir}/{pc_name}/{label_col}'
    os.makedirs(folder_pca, exist_ok=True)

    # Subconjunto: para Family, quita filas sin valor
    if label_col == "Family":
        mask_valid = df["Family"].notna() & (df["Family"].astype(str).str.strip() != "")
    else:
        mask_valid = np.ones(len(df), dtype=bool)

    df_label = df.loc[mask_valid].reset_index(drop=True)
    X_scaled = X_scaled_full[mask_valid]
    X_pca = X_pca_full[mask_valid]

    # Etiquetas categóricas (en el subset)
    labels_cat = df_label[label_col].astype('string').astype('category')
    classes = list(labels_cat.cat.categories)
    y_encoded = labels_cat.cat.codes.to_numpy()

    # Top 12 features por carga en la PC seleccionada
    loadings = np.abs(pca.components_[pc_index])
    top_idx = np.argsort(loadings)[-12:][::-1]
    top_features = [numeric_cols[i] for i in top_idx]

    n_feats = len(top_features)
    grid = math.ceil(math.sqrt(n_feats))

    # Histogramas
    fig_h, axes_h = plt.subplots(grid, grid, figsize=(grid*3, grid*3))
    axes_h = axes_h.flatten()
    for i, feat in enumerate(top_features):
        ax = axes_h[i]
        for cls in classes:
            data = df_label.loc[labels_cat.astype(str) == cls, feat].dropna()
            ax.hist(data, bins=30, alpha=0.5, label=str(cls))
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(n_feats, len(axes_h)):
        fig_h.delaxes(axes_h[j])
    fig_h.suptitle(f'PC{pc_index+1} Histograms by {label_col}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_h.savefig(f'{folder_pca}/histograms_pc{pc_index+1}_{label_col}.pdf')
    plt.close(fig_h)

    # Boxplots
    fig_b, axes_b = plt.subplots(grid, grid, figsize=(grid*3, grid*3))
    axes_b = axes_b.flatten()
    for i, feat in enumerate(top_features):
        ax = axes_b[i]
        data_list = [df_label.loc[labels_cat.astype(str) == cls, feat].dropna() for cls in classes]
        ax.boxplot(data_list, labels=classes, showfliers=True)
        ax.set_title(feat, fontsize=8)
        ax.tick_params(axis='x', labelsize=6, rotation=90)
        ax.tick_params(axis='y', labelsize=6)
    for j in range(n_feats, len(axes_b)):
        fig_b.delaxes(axes_b[j])
    fig_b.suptitle(f'PC{pc_index+1} Boxplots by {label_col}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_b.savefig(f'{folder_pca}/boxplots_pc{pc_index+1}_{label_col}.pdf')
    plt.close(fig_b)

    # PCA scatter
    fig_s = plt.figure(figsize=(6, 6))
    for cls in classes:
        mask = labels_cat.astype(str) == cls
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(cls), s=10, alpha=0.6)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title(f'PCA Scatter PC1 vs PC2 colored by {label_col}', fontsize=12)
    plt.legend(fontsize=6); plt.tight_layout()
    fig_s.savefig(f'{folder_pca}/pca_scatter_pc{pc_index+1}_{label_col}.pdf')
    plt.close(fig_s)

    # ================= UMAP no supervisado (entrenado en el subset) ================
    folder_umap_unsup = f"{base_dir}/umap_unsupervised/{label_col}"

    umap_unsup_reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
    )
    X_umap_unsup = umap_unsup_reducer.fit_transform(X_scaled)

    # --- Regla explícita para Family: quitar puntos con UMAP-1 > UMAP1_MAX ---
    if label_col == "Family":
        mask_keep = X_umap_unsup[:, 0] <= UMAP1_MAX
        # Alinea todo con la máscara
        X_umap_unsup = X_umap_unsup[mask_keep]
        X_scaled = X_scaled[mask_keep]
        X_pca = X_pca[mask_keep]
        df_label = df_label.iloc[mask_keep].reset_index(drop=True)
        labels_cat = labels_cat.iloc[mask_keep].reset_index(drop=True).astype('category')
        classes = list(labels_cat.cat.categories)
        y_encoded = labels_cat.cat.codes.to_numpy()

    # Plot UMAP no supervisado
    fig_u_unsup = plt.figure(figsize=(6, 6))
    for cls in classes:
        mask = labels_cat.astype(str) == cls
        plt.scatter(X_umap_unsup[mask, 0], X_umap_unsup[mask, 1], label=str(cls), s=10, alpha=0.7)
    plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
    plt.title(f'UMAP (unsupervised) colored by {label_col}', fontsize=12)
    plt.legend(fontsize=6); plt.tight_layout()
    fig_u_unsup.savefig(f'{folder_umap_unsup}/umap_unsupervised_{label_col}.pdf')
    plt.close(fig_u_unsup)

    # ================= UMAP supervisado (y codificada) =================
    folder_umap_sup = f"{base_dir}/umap_supervised/{label_col}"
    reducer_sup = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        target_metric='categorical'
    )
    X_umap_sup = reducer_sup.fit_transform(X_scaled, y=y_encoded)

    fig_u_sup = plt.figure(figsize=(6, 6))
    for cls in classes:
        mask = labels_cat.astype(str) == cls
        plt.scatter(X_umap_sup[mask, 0], X_umap_sup[mask, 1], label=str(cls), s=10, alpha=0.7)
    plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
    plt.title(f'UMAP (supervised by {label_col})', fontsize=12)
    plt.legend(fontsize=6); plt.tight_layout()
    fig_u_sup.savefig(f'{folder_umap_sup}/umap_supervised_{label_col}.pdf')
    plt.close(fig_u_sup)

# ------------------------------
# Ejecutar
# ------------------------------
for pc_idx in [0, 1]:
    for label in ['infected', 'Family']:
        generate_plots(pc_idx, label)

print(f"✅ Listo. En 'Family' se eliminaron puntos con UMAP-1 > {UMAP1_MAX} del UMAP no supervisado.")
