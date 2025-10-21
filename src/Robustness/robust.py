#!/usr/bin/env python3
"""
robust.py
---------
Una sola pasada para evaluar robustez de varios modelos introduciendo distintos tipos/porcentajes
de ruido y generando métricas + comparativas de **predicciones** con respecto al dataset original.

Salida (por defecto en ./robustez):
    - metrics.csv
    - metrics_f1macro.csv
    - stability.csv
    - summary_overall.csv
    - summary_by_model.csv
    - f1_macro_summary.csv
    - f1_summary.csv
    - plots/*.pdf
    - plots/combined/*
    - plots/stability/*
    - diffs/*
    - report.md
"""

import argparse
import os
import sys
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- (sin formateo/locator explícito para eje X) ---

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

# Etiquetas para títulos/leyendas
CORR_LABELS: Dict[str, str] = {
    "feat_noise": "Feature noise",
    "label_noise": "Label noise",
    "missing": "Missing values",
    "outliers": "Outliers",
    "dropout_feats": "Feature dropout",
    "none": "None",
}

# --- Estilo y salida de figuras ---
SAVE_EXT = ".pdf"

_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>']
_LINESTYLES = ['-', '--', '-.', ':']

def _plot_with_styles(x_vals, y_vals, idx, label):
    """Serie con estilo consistente + leve jitter horizontal para evitar solapes visuales."""
    marker = _MARKERS[idx % len(_MARKERS)]
    ls = _LINESTYLES[(idx // len(_MARKERS)) % len(_LINESTYLES)]
    jitter = (idx - 2) * 0.002
    x_vals = [x + jitter for x in x_vals]
    plt.plot(x_vals, y_vals, linestyle=ls, marker=marker, label=label, alpha=0.95)

# ---------------------------- Corrupciones ----------------------------

def add_feature_noise(X: pd.DataFrame, level: float, numeric_cols: List[str]) -> pd.DataFrame:
    if level <= 0 or len(numeric_cols) == 0:
        return X
    Xn = X.copy()
    stds = Xn[numeric_cols].std(ddof=0).replace(0, 1.0)
    noise = np.random.normal(loc=0.0, scale=(stds * level).values, size=Xn[numeric_cols].shape)
    Xn[numeric_cols] = Xn[numeric_cols].to_numpy(dtype=float) + noise
    return Xn

def add_missingness(X: pd.DataFrame, level: float) -> pd.DataFrame:
    if level <= 0:
        return X
    Xn = X.copy()
    n_cells = Xn.size
    n_nan = int(level * n_cells)
    if n_nan == 0:
        return Xn
    rows = np.random.randint(0, Xn.shape[0], size=n_nan)
    cols = np.random.randint(0, Xn.shape[1], size=n_nan)
    Xn.values[rows, cols] = np.nan
    return Xn

def add_label_noise(y: pd.Series, level: float) -> pd.Series:
    if level <= 0:
        return y
    y_noisy = y.copy()
    n = len(y_noisy)
    n_flip = int(level * n)
    if n_flip == 0:
        return y_noisy
    idx = np.random.choice(n, size=n_flip, replace=False)
    classes = np.unique(y_noisy)
    for i in idx:
        current = y_noisy.iat[i]
        choices = classes[classes != current]
        if len(choices) > 0:
            y_noisy.iat[i] = np.random.choice(choices)
    return y_noisy

def add_outliers(X: pd.DataFrame, level: float, numeric_cols: List[str]) -> pd.DataFrame:
    if level <= 0 or len(numeric_cols) == 0:
        return X
    Xn = X.copy()
    n_rows = Xn.shape[0]
    n_out = max(1, int(level * n_rows))
    pos = np.random.choice(n_rows, size=n_out, replace=False)
    for col in numeric_cols:
        std = Xn[col].std(ddof=0)
        if not np.isfinite(std) or std == 0:
            std = 1.0
        col_idx = Xn.columns.get_loc(col)
        Xn.iloc[pos, col_idx] = Xn.iloc[pos, col_idx].astype(float) + np.random.normal(0, 5 * std, size=len(pos))
    return Xn

def dropout_features(X: pd.DataFrame, level: float) -> pd.DataFrame:
    if level <= 0:
        return X
    Xn = X.copy()
    n_cols = Xn.shape[1]
    n_drop = max(1, int(level * n_cols))
    cols = np.random.choice(Xn.columns, size=min(n_drop, n_cols), replace=False)
    for c in cols:
        if pd.api.types.is_numeric_dtype(Xn[c]):
            Xn[c] = 0
        else:
            Xn[c] = 'missing'
    return Xn

# ---------------------------- Similitud predicciones ----------------------------

def _safe_normalize(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    proba = np.clip(proba, eps, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba

def js_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = _safe_normalize(P, eps)
    Q = _safe_normalize(Q, eps)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * (np.log(P) - np.log(M)), axis=1)
    kl_qm = np.sum(Q * (np.log(Q) - np.log(M)), axis=1)
    return 0.5 * (kl_pm + kl_qm)

# ---------------------------- Modelos/prepro ----------------------------

def build_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LogReg": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state),
        "GB": GradientBoostingClassifier(random_state=random_state),
        "Ada": AdaBoostClassifier(random_state=random_state),
        "SVC": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=15)
    }

def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot)
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor, numeric_cols, categorical_cols

# ---------------------------- Evaluación ----------------------------

def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    noise_levels: List[float],
    corruptions: List[str],
    n_runs: int,
    test_size: float,
    random_state: int,
    corrupt_train: bool,
    out_dir: str
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots", "combined"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots", "stability"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "diffs"), exist_ok=True)

    preproc, numeric_cols, _ = build_preprocess(X)
    models = build_models(random_state)

    classes_all = np.unique(y)
    is_binary = len(classes_all) == 2

    def _default_pos_label(vals):
        try:
            vals_num = np.array(vals, dtype=float)
            return sorted(vals_num)[-1]
        except Exception:
            return sorted(list(vals))[-1]

    pos_label = _default_pos_label(classes_all) if is_binary else None

    rows_perf, rows_stab = [], []

    for run in range(n_runs):
        rs = random_state + run
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs
        )

        baseline_preds = {}
        baseline_probas = {}

        for model_name, model in models.items():
            clf_base = Pipeline(steps=[("pre", preproc), ("clf", model)])
            clf_base.fit(X_train, y_train)
            y_pred_base = clf_base.predict(X_test)

            proba_base = None
            try:
                if hasattr(clf_base.named_steps["clf"], "predict_proba"):
                    proba_base = clf_base.predict_proba(X_test)
                elif hasattr(clf_base.named_steps["clf"], "decision_function"):
                    dec = clf_base.decision_function(X_test)
                    if dec.ndim == 1:
                        p = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
                        proba_base = np.c_[1 - p, p]
                    else:
                        dec = dec - dec.min(axis=1, keepdims=True)
                        proba_base = dec / (dec.sum(axis=1, keepdims=True) + 1e-9)
            except Exception:
                pass

            f1_normal = (f1_score(y_test, y_pred_base, average="binary", zero_division=0, pos_label=pos_label)
                         if is_binary else
                         f1_score(y_test, y_pred_base, average="macro", zero_division=0))

            metrics_base = {
                "accuracy": accuracy_score(y_test, y_pred_base),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_base),
                "precision_macro": precision_score(y_test, y_pred_base, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred_base, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred_base, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred_base, average="weighted", zero_division=0),
                "f1": f1_normal,
            }
            try:
                if proba_base is not None:
                    if is_binary:
                        if proba_base.ndim == 1 or proba_base.shape[1] == 1:
                            scores = proba_base if proba_base.ndim == 1 else proba_base.ravel()
                            metrics_base["roc_auc"] = roc_auc_score(y_test, scores)
                        else:
                            metrics_base["roc_auc"] = roc_auc_score(y_test, proba_base[:, 1])
                    else:
                        metrics_base["roc_auc_ovr_macro"] = roc_auc_score(
                            y_test, proba_base, multi_class="ovr", average="macro"
                        )
            except Exception:
                pass

            rows_perf.append({
                "run": run, "model": model_name, "corruption": "none", "level": 0.0, **metrics_base
            })

            baseline_preds[model_name] = y_pred_base
            baseline_probas[model_name] = proba_base

        # Corrupciones
        for corr in corruptions:
            for level in noise_levels:
                X_train_c, y_train_c = X_train.copy(), y_train.copy()
                X_test_c, y_test_c = X_test.copy(), y_test.copy()

                def apply_corr(Xdf, Yser):
                    if corr == "feat_noise":
                        return add_feature_noise(Xdf, level, numeric_cols), Yser
                    elif corr == "missing":
                        return add_missingness(Xdf, level), Yser
                    elif corr == "label_noise":
                        return Xdf, add_label_noise(Yser, level)
                    elif corr == "outliers":
                        return add_outliers(Xdf, level, numeric_cols), Yser
                    elif corr == "dropout_feats":
                        return dropout_features(Xdf, level), Yser
                    elif corr == "none":
                        return Xdf, Yser
                    else:
                        raise ValueError(f"Corrupción no soportada: {corr}")

                if corrupt_train:
                    X_train_c, y_train_c = apply_corr(X_train_c, y_train_c)
                X_test_c, y_test_c = apply_corr(X_test_c, y_test_c)

                for model_name, model in models.items():
                    clf = Pipeline(steps=[("pre", preproc), ("clf", model)])
                    clf.fit(X_train_c, y_train_c)

                    y_pred = clf.predict(X_test_c)

                    f1_normal = (f1_score(y_test_c, y_pred, average="binary", zero_division=0, pos_label=pos_label)
                                 if is_binary else
                                 f1_score(y_test_c, y_pred, average="macro", zero_division=0))

                    metrics = {
                        "accuracy": accuracy_score(y_test_c, y_pred),
                        "balanced_accuracy": balanced_accuracy_score(y_test_c, y_pred),
                        "precision_macro": precision_score(y_test_c, y_pred, average="macro", zero_division=0),
                        "recall_macro": recall_score(y_test_c, y_pred, average="macro", zero_division=0),
                        "f1_macro": f1_score(y_test_c, y_pred, average="macro", zero_division=0),
                        "f1_weighted": f1_score(y_test_c, y_pred, average="weighted", zero_division=0),
                        "f1": f1_normal,
                    }

                    proba = None
                    try:
                        if hasattr(clf.named_steps["clf"], "predict_proba"):
                            proba = clf.predict_proba(X_test_c)
                        elif hasattr(clf.named_steps["clf"], "decision_function"):
                            dec = clf.decision_function(X_test_c)
                            if dec.ndim == 1:
                                p = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
                                proba = np.c_[1 - p, p]
                            else:
                                dec = dec - dec.min(axis=1, keepdims=True)
                                proba = dec / (dec.sum(axis=1, keepdims=True) + 1e-9)
                        if proba is not None:
                            if is_binary:
                                if proba.ndim == 1 or proba.shape[1] == 1:
                                    scores = proba if proba.ndim == 1 else proba.ravel()
                                    metrics["roc_auc"] = roc_auc_score(y_test_c, scores)
                                else:
                                    metrics["roc_auc"] = roc_auc_score(y_test_c, proba[:, 1])
                            else:
                                metrics["roc_auc_ovr_macro"] = roc_auc_score(
                                    y_test_c, proba, multi_class="ovr", average="macro"
                                )
                    except Exception:
                        pass

                    rows_perf.append({
                        "run": run, "model": model_name, "corruption": corr, "level": level, **metrics
                    })

                    y_pred_base = baseline_preds[model_name]
                    flip_rate = float(np.mean(y_pred != y_pred_base))
                    agree_rate = 1.0 - flip_rate
                    kappa = cohen_kappa_score(y_pred_base, y_pred)

                    mean_l1, mean_js, mean_conf_drop = np.nan, np.nan, np.nan
                    proba_base = baseline_probas[model_name]
                    if proba_base is not None and proba is not None and proba_base.shape == proba.shape:
                        mean_l1 = float(np.mean(np.sum(np.abs(proba - proba_base), axis=1)))
                        mean_js = float(np.mean(js_divergence(proba_base, proba)))
                        conf_base = np.max(proba_base, axis=1)
                        conf_noisy = np.max(proba, axis=1)
                        mean_conf_drop = float(np.mean(conf_base - conf_noisy))

                    rows_stab.append({
                        "run": run, "model": model_name, "corruption": corr, "level": level,
                        "flip_rate": flip_rate, "agree_rate": agree_rate, "cohen_kappa": kappa,
                        "mean_l1": mean_l1, "mean_js": mean_js, "mean_confidence_drop": mean_conf_drop
                    })

                    # Depuración opcional
                    try:
                        labels_union = np.unique(np.concatenate([y_pred_base, y_pred]))
                        cm = confusion_matrix(y_pred_base, y_pred, labels=labels_union)
                        cm_df = pd.DataFrame(cm,
                                             index=[f"base_{c}" for c in labels_union],
                                             columns=[f"noisy_{c}" for c in labels_union])
                        cm_path = os.path.join(out_dir, "diffs", f"{model_name}_{corr}_{level}_run{run}_flip_matrix.csv")
                        cm_df.to_csv(cm_path, index=True)
                    except Exception:
                        pass

                    try:
                        diffs_df = pd.DataFrame({
                            "y_true": y_test_c.values,
                            "y_pred_base": y_pred_base,
                            "y_pred_noisy": y_pred,
                            "flipped": (y_pred != y_pred_base)
                        })
                        diffs_path = os.path.join(out_dir, "diffs", f"{model_name}_{corr}_{level}_run{run}_preds.csv")
                        diffs_df.to_csv(diffs_path, index=False)
                    except Exception:
                        pass

    # --- Salidas tabulares
    df = pd.DataFrame(rows_perf)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    df[["run", "model", "corruption", "level", "f1_macro"]].to_csv(
        os.path.join(out_dir, "metrics_f1macro.csv"), index=False
    )

    stab = pd.DataFrame(rows_stab)
    stab.to_csv(os.path.join(out_dir, "stability.csv"), index=False)

    summary_by_model = df.groupby(["model", "corruption", "level"]).mean(numeric_only=True).reset_index()
    summary_by_model = summary_by_model.round(3)
    summary_by_model.to_csv(os.path.join(out_dir, "summary_by_model.csv"), index=False)

    summary_overall = df.groupby(["model", "level"]).mean(numeric_only=True).reset_index()
    summary_overall = summary_overall.round(3)
    summary_overall.to_csv(os.path.join(out_dir, "summary_overall.csv"), index=False)

    f1_macro_summary = (
        df.groupby(["model", "corruption", "level"])["f1_macro"]
          .mean()
          .reset_index()
          .sort_values(["model", "corruption", "level"])
    ).round(3)
    f1_macro_summary.to_csv(os.path.join(out_dir, "f1_macro_summary.csv"), index=False)

    clean = (df[(df["corruption"] == "none") & (df["level"] == 0.0)]
             .groupby("model")["f1_macro"].mean().rename("f1_clean"))

    noisy_by_corr = (df[df["corruption"] != "none"]
                     .groupby(["model", "corruption"])["f1_macro"]
                     .mean().unstack("corruption"))

    if noisy_by_corr is not None:
        noisy_by_corr = noisy_by_corr.add_prefix("f1_")

    noisy_overall_mean = (df[df["corruption"] != "none"]
                          .groupby("model")["f1_macro"].mean()
                          .rename("f1_noisy_mean"))

    noisy_overall_min = (df[df["corruption"] != "none"]
                         .groupby("model")["f1_macro"].min()
                         .rename("f1_noisy_min"))

    f1_summary = pd.concat([clean, noisy_by_corr, noisy_overall_mean, noisy_overall_min], axis=1)

    for col in ["f1_feat_noise", "f1_label_noise", "f1_missing", "f1_outliers", "f1_dropout_feats"]:
        if col not in f1_summary.columns:
            f1_summary[col] = np.nan

    f1_summary["degradation_mean"]  = f1_summary["f1_clean"] - f1_summary["f1_noisy_mean"]
    f1_summary["degradation_worst"] = f1_summary["f1_clean"] - f1_summary["f1_noisy_min"]

    ordered_cols = ["f1_clean",
                    "f1_feat_noise", "f1_label_noise", "f1_missing", "f1_outliers", "f1_dropout_feats",
                    "f1_noisy_mean", "degradation_mean", "f1_noisy_min", "degradation_worst"]
    existing_cols = [c for c in ordered_cols if c in f1_summary.columns]
    f1_summary = f1_summary[existing_cols].reset_index().round(3)
    f1_summary.to_csv(os.path.join(out_dir, "f1_summary.csv"), index=False)

    # -------------------------------------------------------------
    # LÍMITES GLOBALES DE Y POR TIPO DE GRÁFICO (X en automático)
    # -------------------------------------------------------------
    def _global_min(series, default):
        s = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return default
        return float(s.min())

    def _global_max(series, default):
        s = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return default
        return float(s.max())

    def _pad_limits(ymin: float, ymax: float, frac: float = 0.04, abs_min_pad: float = 1e-3):
        """Devuelve (ymin_padded, ymax_padded) con pequeño margen relativo y absoluto."""
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return ymin, ymax
        if ymin == ymax:
            pad = max(abs_min_pad, 0.05 * max(1.0, abs(ymax)))
            return ymin - pad, ymax + pad
        rng = ymax - ymin
        pad = max(abs_min_pad, frac * rng)
        return ymin - pad, ymax + pad

    # F1-macro: mantener techo 1.0, suelo inteligente
    F1_YMIN = min(0.5, _global_min(df["f1_macro"], 0.5))

    # flip_rate y JS: rango observado (min..max) con padding
    FR_YMIN = _global_min(stab["flip_rate"], 0.0)
    FR_YMAX = _global_max(stab["flip_rate"], 1.0)
    FR_YMIN, FR_YMAX = _pad_limits(FR_YMIN, FR_YMAX)

    JS_YMIN = _global_min(stab["mean_js"], 0.0)
    JS_YMAX = _global_max(stab["mean_js"], 1.0)
    JS_YMIN, JS_YMAX = _pad_limits(JS_YMIN, JS_YMAX)

    # ==========================
    # Gráficas de RENDIMIENTO (F1) por corrupción/modelo (no combinadas)
    # ==========================
    for corr in df["corruption"].unique():
        corr_label = CORR_LABELS.get(corr, corr)
        for model_name in df["model"].unique():
            sub = df[(df["corruption"] == corr) & (df["model"] == model_name)].groupby("level").mean(numeric_only=True)
            if sub.empty or "f1" not in sub.columns:
                continue
            plt.figure()
            plt.plot(sub.index.to_list(), sub["f1"].to_list(), marker="o")
            plt.title(f"{model_name} - {corr_label} - F1")
            plt.xlabel("Noise level")
            plt.ylabel("F1 score")
            out_pdf = os.path.join(out_dir, "plots", f"{model_name}_{corr}_f1{SAVE_EXT}")
            plt.savefig(out_pdf, bbox_inches="tight")
            plt.close()

    # ==========================
    # COMBINED por MODELO (F1-MACRO) — X automático, Y común
    # ==========================
    combined_dir = os.path.join(out_dir, "plots", "combined")
    gb = df.groupby(["model", "corruption", "level"]).mean(numeric_only=True).reset_index()
    corr_order = ["feat_noise", "label_noise", "missing", "outliers", "dropout_feats", "none"]

    for model_name in gb["model"].unique():
        sub_model = gb[gb["model"] == model_name]
        if sub_model.empty or "f1_macro" not in sub_model.columns:
            continue
        pivot = sub_model.pivot_table(index="level", columns="corruption", values="f1_macro", aggfunc="mean")
        pivot = pivot.sort_index()
        cols_for_metric = [c for c in corr_order if c in pivot.columns and c != "none"]
        pivot = pivot.reindex(columns=cols_for_metric)

        plt.figure()
        for idx, col in enumerate(pivot.columns):
            yvals = pivot[col].to_list()
            _plot_with_styles(pivot.index.to_list(), yvals, idx, CORR_LABELS.get(col, col))
        ax = plt.gca()
        ax.set_title(f"{model_name} - F1-macro vs. noise level")
        ax.set_xlabel("Noise level")
        ax.set_ylabel("F1-macro")
        ax.set_ylim(F1_YMIN, 1.0)  # techo 1.0, suelo inteligente

        plt.legend(title="Corruption type", ncol=2, frameon=True)
        out_pdf = os.path.join(combined_dir, f"{model_name}_combined_f1_macro{SAVE_EXT}")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()

    # ==========================
    # ESTABILIDAD (flip_rate y JS) — X automático, Y = [min,max] observado
    # ==========================
    stab_dir = os.path.join(out_dir, "plots", "stability")
    gb_stab = stab.groupby(["model", "corruption", "level"]).mean(numeric_only=True).reset_index()
    corr_order_stab = ["feat_noise", "label_noise", "missing", "outliers", "dropout_feats"]

    for model_name in gb_stab["model"].unique():
        sub_model = gb_stab[gb_stab["model"] == model_name]
        if sub_model.empty:
            continue

        # flip_rate
        pivot = sub_model.pivot_table(index="level", columns="corruption",
                                      values="flip_rate", aggfunc="mean")
        pivot = pivot.sort_index()
        cols = [c for c in corr_order_stab if c in pivot.columns]
        pivot = pivot.reindex(columns=cols).fillna(np.nan)

        plt.figure()
        for idx, col in enumerate(pivot.columns):
            _plot_with_styles(pivot.index.to_list(), pivot[col].to_list(), idx, CORR_LABELS.get(col, col))
        ax = plt.gca()
        ax.set_title(f"{model_name} - flip_rate vs. noise level")
        ax.set_xlabel("Noise level")
        ax.set_ylabel("flip_rate")
        ax.set_ylim(FR_YMIN, FR_YMAX)  # rango observado con padding

        plt.legend(title="Corruption type", ncol=2, frameon=True)
        out_pdf = os.path.join(stab_dir, f"{model_name}_combined_flip_rate{SAVE_EXT}")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()

        # mean_js
        if "mean_js" in sub_model.columns and sub_model["mean_js"].notna().any():
            pivot = sub_model.pivot_table(index="level", columns="corruption",
                                          values="mean_js", aggfunc="mean")
            pivot = pivot.sort_index()
            cols = [c for c in corr_order_stab if c in pivot.columns]
            pivot = pivot.reindex(columns=cols).fillna(np.nan)

            plt.figure()
            for idx, col in enumerate(pivot.columns):
                _plot_with_styles(pivot.index.to_list(), pivot[col].to_list(), idx, CORR_LABELS.get(col, col))
            ax = plt.gca()
            ax.set_title(f"{model_name} - JS divergence vs. noise level")
            ax.set_xlabel("Noise level")
            ax.set_ylabel("Mean JS")
            ax.set_ylim(JS_YMIN, JS_YMAX)  # rango observado con padding

            plt.legend(title="Corruption type", ncol=2, frameon=True)
            out_pdf = os.path.join(stab_dir, f"{model_name}_combined_js{SAVE_EXT}")
            plt.savefig(out_pdf, bbox_inches="tight")
            plt.close()

    # Informe Markdown
    try:
        top_line = (
            summary_overall
            .sort_values(["f1_macro", "balanced_accuracy", "accuracy"], ascending=False)
            .groupby("model").head(1)
            [["model", "level", "f1_macro", "balanced_accuracy", "accuracy"]]
            .reset_index(drop=True)
        )
        try:
            top_table = top_line.to_markdown(index=False)
        except Exception:
            top_table = top_line.to_string(index=False)
    except Exception:
        top_table = "No se pudo generar el resumen."

    md = ["# Informe de Robustez",
          "",
          "Este informe resume rendimiento y estabilidad de predicciones vs. baseline limpio.",
          "",
          "## Mejor rendimiento medio por modelo (nivel con mayor F1-macro):", "",
          top_table,
          "",
          "## Archivos clave",
          "- *metrics.csv*: rendimiento vs. y_true (incluye f1_macro y f1).",
          "- *metrics_f1macro.csv*: f1_macro por run/model/corr/level.",
          "- *stability.csv*: flip_rate, kappa, L1, JS, drop de confianza.",
          "- *diffs/*: matrices de cambio y comparativas por muestra.",
          "- *plots/*.pdf*: curvas por corrupción (F1).",
          "- *plots/combined/*: rendimiento combinado por modelo (F1-macro).",
          "- *plots/stability/*: estabilidad (flip_rate, JS) por modelo.",
          "- *f1_macro_summary.csv*, *f1_summary.csv*.",
          ]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Ruta al CSV con los datos")
    parser.add_argument("--target", type=str, required=True, help="Nombre de la columna objetivo")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--corruptions", type=str, nargs="+",
                        default=["feat_noise", "label_noise", "missing", "outliers", "dropout_feats"],
                        choices=["none", "feat_noise", "label_noise", "missing", "outliers", "dropout_feats"])
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--corrupt_train", action="store_true")
    parser.add_argument("--out_dir", type=str, default="robustez")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: No existe el fichero {args.data}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        print(f"ERROR: La columna objetivo '{args.target}' no está en el dataset.", file=sys.stderr)
        sys.exit(1)

    y = df[args.target]
    X = df.drop(columns=[args.target])

    _ = evaluate_models(
        X=X, y=y,
        noise_levels=args.noise_levels,
        corruptions=args.corruptions,
        n_runs=args.n_runs,
        test_size=args.test_size,
        random_state=args.random_state,
        corrupt_train=args.corrupt_train,
        out_dir=args.out_dir
    )

    print(f"✓ Pruebas completadas. Reportes en: {args.out_dir}")
    print(f"- {args.out_dir}/metrics.csv")
    print(f"- {args.out_dir}/metrics_f1macro.csv")
    print(f"- {args.out_dir}/stability.csv")
    print(f"- {args.out_dir}/summary_by_model.csv")
    print(f"- {args.out_dir}/summary_overall.csv")
    print(f"- {args.out_dir}/f1_macro_summary.csv")
    print(f"- {args.out_dir}/f1_summary.csv")
    print(f"- {args.out_dir}/report.md")

if __name__ == "__main__":
    main()
