#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explicabilidad global y local (LogReg, RF, GB, Ada, SVC, KNN)
Genera: perm_importance, native/coefs, PDP/ICE (solo RF/GB con features pedidas),
        SHAP (meanabs, beeswarm, 3 waterfalls)
Salida vectorial: SVG y PDF para todas las figuras
Además: Collages 3x2 en PDF (y PNG espejo) para:
        - 6 permutation importances
        - 6 beeswarm
        - 6 waterfalls (primera disponible por modelo)

Objetivo de uniformidad:
- Mismo TAMAÑO del RECUDRO del gráfico (eje principal) en TODAS las figuras.
- Si hay etiquetas largas, se añade blanco a la izquierda (figura total puede crecer),
  pero el recuadro del gráfico queda idéntico en tamaño y posición.
"""

import argparse, os, sys, warnings, glob
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# SHAP opcional
try:
    import shap
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Preferencias globales Matplotlib (mejor texto vectorial)
# -----------------------------------------------------------------------------
plt.rcParams["svg.fonttype"] = "none"      # texto como texto (no curvas) en SVG
plt.rcParams["pdf.fonttype"] = 42          # fuentes TrueType en PDF (editable)
# MUY IMPORTANTE: no recortar al contenido; mantener tamaño fijo de figura
plt.rcParams["savefig.bbox"] = "standard"

# === UNIFORM FIGSIZE & MARGINS (se recalculan en main) =======================
# Objetivo: MISMO tamaño del recuadro del gráfico (eje principal) en pulgadas,
# y figura final única (la "más grande") añadiendo blanco a la izquierda si hace falta.
_TARGET_FIGSIZE: Tuple[float, float] = (8.0, 5.0)            # se recalcula
_AX_RECT: Tuple[float, float, float, float] = (0.30, 0.12, 0.65, 0.80)  # l,b,w,h (fracciones), se recalcula
_AX_TARGET_IN: Tuple[float, float] = (6.2, 4.2)              # tamaño deseado del RECUDRO (eje) en pulgadas

def _compute_target_figsize_from_rect(ax_in: Tuple[float,float],
                                      rect: Tuple[float,float,float,float]) -> Tuple[float,float]:
    """Devuelve figsize (W,H) tal que el eje tenga 'ax_in' pulgadas, dado rect=(l,b,w,h) en fracciones."""
    l, b, w, h = rect
    W_fig = ax_in[0] / max(w, 1e-6)
    H_fig = ax_in[1] / max(h, 1e-6)
    return (W_fig, H_fig)

def _compute_margins_from_max_label(max_len: int) -> Tuple[float,float,float,float]:
    """
    Márgenes fraccionales globales basados en la etiqueta Y más larga GLOBAL:
    - 'left' crece con max_len para no cortar etiquetas.
    - 'right' reserva algo para colorbar de SHAP beeswarm.
    NOTA: Usamos estos márgenes para TODAS las figuras para que el recuadro
    del eje (en pulgadas) sea constante; si left es grande, se añade blanco.
    """
    left   = min(0.60, 0.20 + 0.009 * max_len)  # generoso con nombres largos
    right  = 0.08
    bottom = 0.12
    top    = 0.06
    width  = max(0.20, 1.0 - left - right)
    height = 1.0 - bottom - top
    return (left, bottom, width, height)

def _apply_axes_rect(ax):
    """Fuerza el mismo rectángulo de ejes (inner plot) en todas las figuras."""
    try:
        ax.set_position(_AX_RECT)
    except Exception:
        pass

def _clean_labels(labels: List[str]) -> List[str]:
    """Limpia etiquetas (quita prefijos 'num_' / 'num__')."""
    out = []
    for l in labels:
        s = str(l)
        if s.startswith("num__"):
            s = s[len("num__"):]
        if s.startswith("num_"):
            s = s[len("num_"):]
        out.append(s)
    return out

def _keep_mask_not_nan(feat_names: List[str]) -> np.ndarray:
    """True para nombres que no sean exactamente 'nan' tras limpieza."""
    clean = _clean_labels(feat_names)
    return np.array([str(s).strip().lower() != "nan" for s in clean], dtype=bool)

# --------- Modelos / prepro ----------
def build_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LogReg": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state),
        "GB": GradientBoostingClassifier(random_state=random_state),
        "Ada": AdaBoostClassifier(random_state=random_state, algorithm="SAMME"),
        "SVC": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=15)
    }

def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")
    return pre, num_cols, cat_cols

def get_feature_names(pre: ColumnTransformer) -> List[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                try:
                    names.extend(list(trans.get_feature_names_out(cols)))
                    continue
                except Exception:
                    pass
            names.extend(list(cols if isinstance(cols, (list, tuple)) else [cols]))
        return names

# --------- Guardado ----------
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _save_vector(out_base: str, fig=None):
    """Guarda SVG, PDF y PNG (300 dpi) sin recorte para tamaño constante."""
    if fig is None:
        fig = plt.gcf()
    svg_path = f"{out_base}.svg"
    pdf_path = f"{out_base}.pdf"
    png_path = f"{out_base}.png"
    ensure_dir(svg_path)
    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"  · guardado: {svg_path}")
    print(f"  · guardado: {pdf_path}")
    print(f"  · guardado: {png_path}")

# --------- Collages 3x2 ----------
def _compose_3x2(image_paths, titles, out_pdf_path, page_title=""):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 apaisado aprox.
    for i, (p, t) in enumerate(zip(image_paths, titles), start=1):
        ax = fig.add_subplot(2, 3, i)
        ax.axis("off")
        if p and os.path.exists(p):
            try:
                img = plt.imread(p)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "(no disponible)", ha="center", va="center", fontsize=10)
        else:
            ax.text(0.5, 0.5, "(no disponible)", ha="center", va="center", fontsize=10)
        ax.set_title(t, fontsize=10)
    if page_title:
        fig.suptitle(page_title, fontsize=14)
    ensure_dir(out_pdf_path)
    fig.savefig(out_pdf_path, format="pdf")
    fig.savefig(out_pdf_path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"  · collage guardado: {out_pdf_path}")
    print(f"  · collage guardado: {out_pdf_path.replace('.pdf', '.png')}")

def _find_first(patterns):
    for pat in patterns:
        found = sorted(glob.glob(pat))
        if found:
            return found[0]
    return None

def _collage_for_kind(out_dir: str, kind: str):
    order = [("Ada", "ADA"), ("GB", "GB"), ("KNN", "KNN"),
             ("LogReg", "LOGREG"), ("RF", "RF"), ("SVC", "SVC")]
    images, titles = [], []
    for name, title in order:
        if kind == "perm":
            base = os.path.join(out_dir, name, f"{name}_perm_importance_f1macro")
            p = f"{base}.png"
        elif kind == "bee":
            base = os.path.join(out_dir, name, f"{name}_shap_beeswarm")
            p = f"{base}.png"
        elif kind == "water":
            local_dir = os.path.join(out_dir, name, "local")
            p = _find_first([
                os.path.join(local_dir, f"{name}_waterfall_idx0_0.png"),
                os.path.join(local_dir, f"{name}_waterfall_idx*_0.png"),
                os.path.join(local_dir, f"{name}_waterfall_*.png"),
            ])
        else:
            p = None
        images.append(p if (p and os.path.exists(p)) else None)
        titles.append(title)

    os.makedirs(os.path.join(out_dir, "collages"), exist_ok=True)
    if kind == "perm":
        outp = os.path.join(out_dir, "collages", "collage_permutation_importances.pdf")
        title = "Permutation Importances (F1-macro) — ADA, GB, KNN / LOGREG, RF, SVC"
    elif kind == "bee":
        outp = os.path.join(out_dir, "collages", "collage_beeswarm.pdf")
        title = "SHAP Beeswarm (ADA, GB, KNN / LOGREG, RF, SVC)"
    else:
        outp = os.path.join(out_dir, "collages", "collage_waterfalls.pdf")
        title = "SHAP Waterfalls (ADA, GB, KNN / LOGREG, RF, SVC)"
    _compose_3x2(images, titles, outp, page_title=title)

def make_collages(out_dir: str):
    print("Creando collages 3x2…")
    _collage_for_kind(out_dir, "perm")
    _collage_for_kind(out_dir, "bee")
    _collage_for_kind(out_dir, "water")

# --------- Utilidades SHAP / figuras ----------
def _largest_axes(fig: plt.Figure):
    """Devuelve el eje de mayor área dentro de una figura (para SHAP, etc.)."""
    if not fig.axes:
        return None
    ax_areas = [(ax, ax.get_position().width * ax.get_position().height) for ax in fig.axes]
    ax_areas.sort(key=lambda t: t[1], reverse=True)
    return ax_areas[0][0]

from contextlib import contextmanager
@contextmanager
def _uniform_figsize_ctx():
    """
    Fuerza el mismo tamaño de figura para trazados que crean su propia Figure
    (e.g., shap.plots.waterfall/beeswarm en algunas versiones).
    """
    with plt.rc_context({"figure.figsize": _TARGET_FIGSIZE}):
        yield

# --------- Plots base ----------
def save_bar(values: np.ndarray, labels: List[str], title: str, out_base: str,
             top_k: int = 20, xlabel: str = None):
    labels_clean = _clean_labels(labels)
    mask = np.array([str(s).strip().lower() != "nan" for s in labels_clean], dtype=bool)
    values = np.asarray(values)[mask]
    labels_clean = np.asarray(labels_clean)[mask]

    order = np.argsort(np.abs(values))[::-1][:top_k]

    fig = plt.figure(figsize=_TARGET_FIGSIZE)  # figura con tamaño común
    plt.barh(labels_clean[order][::-1], np.asarray(values)[order][::-1])
    plt.title(title)
    plt.xlabel(xlabel or "Importance / Coefficient / Mean |SHAP|")
    _apply_axes_rect(plt.gca())
    _save_vector(out_base, fig=fig)

def plot_permutation_importance(clf: Pipeline, X_test, y_test, feat_names, out_dir, model_name, top_k):
    res = permutation_importance(
        clf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring=make_scorer(f1_score, average="macro")
    )
    out_base = os.path.join(out_dir, model_name, f"{model_name}_perm_importance_f1macro")
    save_bar(
        res.importances_mean,
        feat_names,
        f"{model_name} - Permutation Importance (F1-macro)",
        out_base,
        top_k=top_k,
        xlabel="Mean decrease in F1-macro (permutation importance)"
    )
    mask_keep = _keep_mask_not_nan(list(feat_names))
    names_clean = np.array(_clean_labels(list(feat_names)))[mask_keep]
    means = res.importances_mean[mask_keep]
    stds  = res.importances_std[mask_keep]
    df_pi = pd.DataFrame({
        "feature": names_clean,
        "mean_decrease_f1_macro": means,
        "std_decrease_f1_macro": stds,
        "n_repeats": res.importances.shape[1]
    }).sort_values("mean_decrease_f1_macro", ascending=False)
    csv_path = f"{out_base}.csv"
    ensure_dir(csv_path)
    df_pi.to_csv(csv_path, index=False)
    print(f"  · guardado: {csv_path}")

def plot_native_importance_or_coefs(model, feat_names, out_dir, model_name, top_k):
    vals = None
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        xlabel = "Mean decrease in impurity (MDI)"
    elif hasattr(model, "coef_"):
        vals = np.mean(np.abs(model.coef_), axis=0)
        xlabel = "Absolute coefficient (standardized features)"
    else:
        xlabel = None
    if vals is not None and len(vals) == len(feat_names):
        out_base = os.path.join(out_dir, model_name, f"{model_name}_native_importance")
        save_bar(vals, feat_names, f"{model_name} - Native importance / coefficients", out_base, top_k=top_k, xlabel=xlabel)

# --------- PDP/ICE específicos (RF/GB) ----------
def plot_pdp_specific(clf: Pipeline, X_test: pd.DataFrame, out_dir: str, model_name: str):
    targets = []
    if model_name == "RF":
        targets = ["num_FileCloses", "num_NtSetInformationThread"]
    elif model_name == "GB":
        targets = ["num_NtWaitForMultipleObjects", "num_InitializeCriticalSection"]
    else:
        return

    for feat in targets:
        if feat not in X_test.columns:
            print(f"  · aviso: '{feat}' no está en X_test, omito PDP/ICE.")
            continue
        feat_clean = _clean_labels([feat])[0]
        try:
            fig, ax = plt.subplots(figsize=_TARGET_FIGSIZE)
            PartialDependenceDisplay.from_estimator(
                clf, X_test, [feat], kind="both", ax=ax,
                ice_lines_kw={"alpha": 0.2}, pd_line_kw={"linestyle": "--"}
            )
            ax.set_title(f"{model_name} - PDP/ICE: {feat_clean}")
            ax.set_xlabel("Feature value (X-axis)")
            ax.set_ylabel("Partial dependence (Y-axis)")
            _apply_axes_rect(ax)
            out_base = os.path.join(out_dir, model_name, "pdp", f"{model_name}_pdp_{feat_clean}")
            _save_vector(out_base, fig=fig)
        except Exception as e:
            print(f"  · PDP/ICE falló en {feat}: {e}")

# --------- SHAP helpers ----------
def _pick_positive_class_index(est) -> Optional[int]:
    if hasattr(est, "classes_") and len(est.classes_) == 2:
        classes = est.classes_
        for cand in (1, True, "1", "true", "True", "positive", "pos", "malware", "ransomware"):
            idxs = np.where(np.array(classes, dtype=object) == cand)[0]
            if len(idxs): return int(idxs[0])
        return 1
    return None

def _explain_to_2d(values, base_values):
    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            return values, base_values
        if values.ndim == 3:
            if values.shape[0] < values.shape[-1]:   # [C,N,F]
                arr = np.abs(values)
                cls = int(np.argmax(arr.mean(axis=(1,2))))
                return values[cls], base_values
            else:                                     # [N,F,C]
                arr = np.abs(values)
                cls = int(np.argmax(arr.mean(axis=(0,1))))
                return values[:,:,cls], base_values
    return values, base_values

def _to_explanation(values_2d: np.ndarray, base_value, data_matrix: np.ndarray, feature_names: List[str]) -> "shap.Explanation":
    base = base_value
    if isinstance(base_value, (list, np.ndarray)) and np.size(base_value) > 1:
        base = np.mean(base_value)
    return shap.Explanation(values=values_2d, base_values=base, data=data_matrix, feature_names=feature_names)

def _bar_from_shap_any(sv, feat_names, out_base, title, top_k):
    feat_names = list(feat_names)
    mask_keep = _keep_mask_not_nan(feat_names)
    feat_names_clean = np.array(_clean_labels(feat_names))[mask_keep]
    try:
        if isinstance(sv, list):
            arr = np.stack([np.abs(a) for a in sv], axis=0)  # [C,N,F]
            mean_abs = arr.mean(axis=(0,1))                  # [F]
        elif hasattr(sv, "values"):
            vals2d, _ = _explain_to_2d(sv.values, getattr(sv, "base_values", 0.0))  # [N,F]
            mean_abs = np.mean(np.abs(vals2d), axis=0)       # [F]
        elif isinstance(sv, np.ndarray):
            vals2d, _ = _explain_to_2d(sv, 0.0)              # [N,F]
            mean_abs = np.mean(np.abs(vals2d), axis=0)       # [F]
        else:
            return
        mean_abs = np.asarray(mean_abs)[mask_keep]
        save_bar(mean_abs, list(feat_names_clean), title, out_base, top_k=top_k, xlabel="Mean |SHAP| value")
    except Exception as e:
        print(f"    (aviso) no se pudo guardar barra |SHAP|: {e}")

def _make_beeswarm(sv, X_sm, feat_names, out_base: str, est=None, multiclass_extra: bool = False):
    feat_names = list(feat_names)
    mask_keep = _keep_mask_not_nan(feat_names)
    feat_names_clean = np.array(_clean_labels(feat_names))[mask_keep]

    def _subset_vals(vals2d):
        return vals2d[:, mask_keep]

    X_sm_f = X_sm[:, mask_keep] if (isinstance(X_sm, np.ndarray) and X_sm.ndim == 2) else X_sm

    try:
        if hasattr(sv, "values"):
            vals2d, base = _explain_to_2d(sv.values, getattr(sv, "base_values", 0.0))  # [N,F]
            vals2d = _subset_vals(vals2d)
            exp = _to_explanation(vals2d, base, X_sm_f, list(feat_names_clean))
            with _uniform_figsize_ctx():
                shap.plots.beeswarm(exp, show=False)
                fig = plt.gcf()
                ax = _largest_axes(fig) or plt.gca()
                _apply_axes_rect(ax)
                _save_vector(out_base, fig=fig)
            return

        if isinstance(sv, list):
            if len(sv) == 2:
                pos_idx = _pick_positive_class_index(est)
                chosen = sv[pos_idx if pos_idx is not None else 1]  # [N,F]
                chosen = _subset_vals(chosen)
                exp = _to_explanation(chosen, 0.0, X_sm_f, list(feat_names_clean))
                with _uniform_figsize_ctx():
                    shap.plots.beeswarm(exp, show=False)
                    fig = plt.gcf()
                    ax = _largest_axes(fig) or plt.gca()
                    _apply_axes_rect(ax)
                    _save_vector(out_base, fig=fig)
            else:
                arr = np.stack([np.abs(a) for a in sv], axis=0)     # [C,N,F]
                cls = int(np.argmax(arr.mean(axis=(1,2))))
                chosen = _subset_vals(sv[cls])
                exp = _to_explanation(chosen, 0.0, X_sm_f, list(feat_names_clean))
                with _uniform_figsize_ctx():
                    shap.plots.beeswarm(exp, show=False)
                    fig = plt.gcf()
                    ax = _largest_axes(fig) or plt.gca()
                    _apply_axes_rect(ax)
                    _save_vector(out_base, fig=fig)
                if multiclass_extra:
                    for k, a in enumerate(sv):
                        expk = _to_explanation(_subset_vals(a), 0.0, X_sm_f, list(feat_names_clean))
                        out_k = f"{out_base}_class{k}"
                        with _uniform_figsize_ctx():
                            shap.plots.beeswarm(expk, show=False)
                            fig = plt.gcf()
                            ax = _largest_axes(fig) or plt.gca()
                            _apply_axes_rect(ax)
                            _save_vector(out_k, fig=fig)
        elif isinstance(sv, np.ndarray):
            vals2d, _ = _explain_to_2d(sv, 0.0)  # [N,F]
            vals2d = _subset_vals(vals2d)
            exp = _to_explanation(vals2d, 0.0, X_sm_f, list(feat_names_clean))
            with _uniform_figsize_ctx():
                shap.plots.beeswarm(exp, show=False)
                fig = plt.gcf()
                ax = _largest_axes(fig) or plt.gca()
                _apply_axes_rect(ax)
                _save_vector(out_base, fig=fig)
    except Exception as e:
        print(f"    (aviso) beeswarm no creado: {e}")

def _make_waterfalls(sv, X_sm, feat_names, out_dir: str, model_name: str, picks: List[int], top_k: int = 10):
    feat_names = list(feat_names)
    mask_keep = _keep_mask_not_nan(feat_names)
    feat_names_clean = np.array(_clean_labels(feat_names))[mask_keep]

    if hasattr(sv, "values"):
        vals2d, _ = _explain_to_2d(sv.values, getattr(sv, "base_values", 0.0))  # [N,F]
    elif isinstance(sv, list):
        if len(sv) == 2:
            vals2d = sv[1]  # [N,F]
        else:
            arr = np.stack([np.abs(a) for a in sv], axis=0)
            cls = int(np.argmax(arr.mean(axis=(1,2))))
            vals2d = sv[cls]                                    # [N,F]
    else:
        vals2d, _ = _explain_to_2d(sv, 0.0)                    # [N,F]

    vals2d = vals2d[:, mask_keep]
    X_sm_f = X_sm[:, mask_keep] if (isinstance(X_sm, np.ndarray) and X_sm.ndim == 2) else X_sm

    n = vals2d.shape[0]
    for k, pos in enumerate(picks[:3]):
        if pos < 0 or pos >= n:
            continue
        try:
            exp = _to_explanation(vals2d[pos], 0.0, X_sm_f[pos], list(feat_names_clean))
            with _uniform_figsize_ctx():
                shap.plots.waterfall(exp, show=False, max_display=top_k)
                fig = plt.gcf()
                ax = _largest_axes(fig) or plt.gca()
                _apply_axes_rect(ax)
                out_base = os.path.join(out_dir, model_name, "local", f"{model_name}_waterfall_idx{pos}_{k}")
                _save_vector(out_base, fig=fig)
        except Exception:
            order = np.argsort(np.abs(vals2d[pos]))[::-1][:top_k]
            fig = plt.figure(figsize=_TARGET_FIGSIZE)
            plt.barh(np.array(feat_names_clean)[order][::-1], np.array(vals2d[pos])[order][::-1])
            plt.title(f"{model_name} - SHAP local (idx={pos})")
            _apply_axes_rect(plt.gca())
            out_base = os.path.join(out_dir, model_name, "local", f"{model_name}_localbar_idx{pos}_{k}")
            _save_vector(out_base, fig=fig)

# --------- SHAP por tipo de modelo ----------
def _tree_explainer(est, X_bg, X_sm):
    explainer = shap.TreeExplainer(est, data=X_bg, feature_perturbation="interventional", model_output="probability")
    try:
        sv = explainer.shap_values(X_sm, check_additivity=False)  # lista por clase (API vieja)
        base = explainer.expected_value
        return sv, base
    except TypeError:
        exp = explainer(X_sm, check_additivity=False)             # Explanation (API nueva)
        return exp, getattr(explainer, "expected_value", None)

def _linear_explainer(est, X_bg, X_sm):
    explainer = shap.LinearExplainer(est, X_bg)
    try:
        sv = explainer.shap_values(X_sm)
        base = explainer.expected_value
        return sv, base
    except Exception:
        exp = explainer(X_sm)
        return exp, getattr(explainer, "expected_value", None)

def _kernel_explainer(predict_f, X_bg, X_sm, nsamples=100):
    try:
        X_bg_sum = shap.kmeans(X_bg, min(50, X_bg.shape[0]))
    except Exception:
        X_bg_sum = X_bg
    explainer = shap.KernelExplainer(predict_f, X_bg_sum)
    try:
        sv = explainer.shap_values(X_sm, nsamples=nsamples)
        base = explainer.expected_value
        return sv, base
    except Exception:
        exp = explainer(X_sm, nsamples=nsamples)
        return exp, getattr(explainer, "expected_value", None)

def shap_global_and_local(
    clf: Pipeline, X_train, X_test, y_test,
    out_dir: str, model_name: str, top_k: int,
    rng: np.random.Generator,
    background_size: int = 200, sample_size: int = 300,
    extra_multiclass_beeswarm: bool = False
):
    if not _HAVE_SHAP:
        print(f"[{model_name}] SHAP no disponible, omito SHAP."); return

    pre = clf.named_steps["pre"]; est = clf.named_steps["clf"]
    X_train_t = pre.transform(X_train); X_test_t = pre.transform(X_test)
    feat_names = get_feature_names(pre)

    bg_idx = np.asarray(rng.choice(X_train_t.shape[0], size=min(background_size, X_train_t.shape[0]), replace=False), dtype=int)
    sm_idx = np.asarray(rng.choice(X_test_t.shape[0],  size=min(sample_size,  X_test_t.shape[0]),  replace=False), dtype=int)
    X_bg = X_train_t[bg_idx]; X_sm = X_test_t[sm_idx]

    try:
        if isinstance(est, (RandomForestClassifier, GradientBoostingClassifier)):
            sv, base = _tree_explainer(est, X_bg, X_sm)
        elif isinstance(est, LogisticRegression):
            sv, base = _linear_explainer(est, X_bg, X_sm)
        else:
            predict_f = (lambda Z: est.predict_proba(Z)) if hasattr(est, "predict_proba") else (lambda Z: est.decision_function(Z))
            sv, base = _kernel_explainer(predict_f, X_bg, X_sm, nsamples=80)
    except Exception as e:
        print(f"[{model_name}] SHAP error: {e}"); return

    out_base = os.path.join(out_dir, model_name, f"{model_name}_shap_meanabs")
    _bar_from_shap_any(sv, feat_names, out_base, f"{model_name} - Mean |SHAP| (global)", top_k)

    # >>> línea corregida (sin '}')
    out_bee = os.path.join(out_dir, model_name, f"{model_name}_shap_beeswarm")
    _make_beeswarm(sv, X_sm, feat_names, out_bee, est=est, multiclass_extra=extra_multiclass_beeswarm)

    try:
        _ = clf.predict(X_test)  # para asegurar pipeline
        picks = list(range(min(3, X_sm.shape[0])))
        _make_waterfalls(sv, X_sm, feat_names, out_dir, model_name, picks=picks, top_k=top_k)
    except Exception as e:
        print(f"[{model_name}] No se pudieron generar waterfalls: {e}")

# --------- Main ----------
def main():
    global _TARGET_FIGSIZE, _AX_RECT, _AX_TARGET_IN

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--target", required=True, type=str)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--background_size", type=int, default=200)
    ap.add_argument("--sample_size", type=int, default=300)
    ap.add_argument("--out_dir", type=str, default="explicabilidad")
    ap.add_argument("--ignore_cols", type=str, default="id,Family",
                    help="Columnas a excluir del entrenamiento, separadas por comas (por defecto: id,Family)")
    ap.add_argument("--extra_multiclass_beeswarm", action="store_true",
                    help="Si hay multiclase, además del beeswarm principal guarda uno por clase.")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: no existe {args.data}", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        print(f"ERROR: '{args.target}' no está en el dataset.", file=sys.stderr); sys.exit(1)

    # Excluir columnas (id/Family), insensible a may/min
    ignore = [c.strip() for c in args.ignore_cols.split(",") if c.strip()]
    ignore_lower = {c.lower() for c in ignore}
    cols_to_drop = [c for c in df.columns if c.lower() in ignore_lower and c != args.target]
    if cols_to_drop:
        print(f"Excluyendo columnas del entrenamiento: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    y = df[args.target]; X = df.drop(columns=[args.target])

    # === Calcular FIGSIZE y MÁRGENES UNIFORMES antes de entrenar =================
    # 1) estimar longitud MÁXIMA de etiqueta (global) tras el preprocesado
    pre_tmp, _, _ = build_preprocess(X)
    try:
        pre_tmp.fit(X)  # expandir nombres
        feat_names_all = get_feature_names(pre_tmp)
    except Exception:
        feat_names_all = list(X.columns)
    clean_all = [s for s in _clean_labels(feat_names_all) if str(s).strip().lower() != "nan"]
    max_len = max((len(s) for s in clean_all), default=20)

    # 2) márgenes fraccionales comunes para TODOS los plots (basados en max_len)
    _AX_RECT = _compute_margins_from_max_label(max_len)

    # 3) FIGSIZE tal que el RECUDRO del eje tenga tamaño constante en pulgadas.
    #    Si 'left' es grande por etiquetas, la FIGURE crece -> añade blanco a la izquierda.
    _TARGET_FIGSIZE = _compute_target_figsize_from_rect(_AX_TARGET_IN, _AX_RECT)

    print(f"[AX RECT] (l,b,w,h): {_AX_RECT}")
    print(f"[FIGSIZE] objetivo (W,H) en pulgadas: {_TARGET_FIGSIZE} para eje de {_AX_TARGET_IN} in")

    # ---------------------------------------------------------------------------

    pre, num_cols, _ = build_preprocess(X)
    models = build_models(args.random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.random_state)

    for model_name, model in models.items():
        print(f"[{model_name}] Entrenando y generando explicabilidad...")
        clf = Pipeline([("pre", pre), ("clf", model)])
        clf.fit(X_train, y_train)

        # Carpeta del modelo
        model_dir = os.path.join(args.out_dir, model_name); os.makedirs(model_dir, exist_ok=True)

        # Métricas + matriz de confusión
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        with open(os.path.join(model_dir, f"{model_name}_scores.txt"), "w", encoding="utf-8") as f:
            f.write(f"accuracy={acc:.4f}\n"); f.write(f"f1_macro={f1m:.4f}\n")
        cm = confusion_matrix(y_test, y_pred)
        cm_base = os.path.join(model_dir, f"{model_name}_confusion_matrix")
        ensure_dir(cm_base + ".svg")
        pd.DataFrame(cm).to_csv(cm_base + ".csv", index=False)
        print(f"  · guardado: {cm_base}.csv")

        feat_names = get_feature_names(pre)

        # Global: Permutation Importance (F1-macro)
        try:
            plot_permutation_importance(clf, X_test, y_test, feat_names, args.out_dir, model_name, args.top_k)
        except Exception as e:
            print(f"[{model_name}] Permutation Importance falló: {e}")

        # Nativo o coeficientes
        try:
            plot_native_importance_or_coefs(clf.named_steps["clf"], feat_names, args.out_dir, model_name, args.top_k)
        except Exception:
            pass

        # PDP/ICE SOLO para RF y GB
        plot_pdp_specific(clf, X_test, args.out_dir, model_name)

        # SHAP global y local
        shap_global_and_local(
            clf, X_train, X_test, y_test,
            args.out_dir, model_name, args.top_k,
            rng=rng, background_size=args.background_size, sample_size=args.sample_size,
            extra_multiclass_beeswarm=args.extra_multiclass_beeswarm
        )

    # --- collages 3x2 ---
    make_collages(args.out_dir)

    print(f"✓ Explicabilidad generada en: {args.out_dir}")
    print("Incluye versiones vectoriales (.svg y .pdf) de: permutation importance (Mean decrease in F1-macro), native_importance (si aplica),")
    print("SHAP: *_shap_meanabs, *_shap_beeswarm, local/*; PDP/ICE: RF(FileCloses, NtSetInformationThread) y GB(NtWaitForMultipleObjects, InitializeCriticalSection).")
    print("Además: collages en explicabilidad/collages/*.pdf (y .png).")

if __name__ == "__main__":
    main()
