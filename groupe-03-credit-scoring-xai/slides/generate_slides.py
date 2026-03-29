"""
Génération des slides de présentation — Credit Scoring avec IA Explicable
ECE Paris — Ing4 Finance — Groupe 03 — MALAK El Idrissi & Joe Boueri
Soutenance : 30 mars 2026
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "presentation.pdf")

# ─── Couleurs ──────────────────────────────────────────────────────────────────
C_DARK   = "#1a2332"   # fond foncé
C_BLUE   = "#2563eb"   # bleu ECE
C_LIGHT  = "#eff6ff"   # fond clair
C_ACCENT = "#f59e0b"   # orange accent
C_GREEN  = "#16a34a"
C_RED    = "#dc2626"
C_GRAY   = "#6b7280"
WHITE    = "#ffffff"

FIG_W, FIG_H = 13.33, 7.5   # 16:9


def new_fig():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(C_DARK)
    return fig


def header_bar(fig, title, subtitle=""):
    """Bande de titre en haut."""
    ax = fig.add_axes([0, 0.82, 1, 0.18])
    ax.set_facecolor(C_BLUE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.62, title, ha='center', va='center',
            fontsize=26, fontweight='bold', color=WHITE, fontfamily='DejaVu Sans')
    if subtitle:
        ax.text(0.5, 0.18, subtitle, ha='center', va='center',
                fontsize=13, color="#bfdbfe", fontfamily='DejaVu Sans')


def footer(fig, slide_num, total=15):
    ax = fig.add_axes([0, 0, 1, 0.045])
    ax.set_facecolor(C_BLUE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.5, "ECE Paris — Ing4 Finance — Credit Scoring XAI — Groupe 03",
            ha='center', va='center', fontsize=9, color="#bfdbfe")
    ax.text(0.97, 0.5, f"{slide_num}/{total}",
            ha='right', va='center', fontsize=9, color=WHITE)


def content_area(fig, x=0.04, y=0.055, w=0.92, h=0.755):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(C_DARK)
    ax.axis('off')
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Titre
# ══════════════════════════════════════════════════════════════════════════════
def slide_titre(pdf):
    fig = new_fig()

    # Fond dégradé simulé avec un rectangle
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_DARK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # Bande bleue centrale
    rect = mpatches.FancyBboxPatch((0.07, 0.28), 0.86, 0.46,
                                    boxstyle="round,pad=0.01",
                                    linewidth=0, facecolor=C_BLUE, alpha=0.9)
    ax.add_patch(rect)

    ax.text(0.5, 0.87, "Groupe 03 — ECE Paris, Ing4 Finance",
            ha='center', fontsize=12, color="#93c5fd", style='italic')
    ax.text(0.5, 0.64, "Credit Scoring avec IA Explicable",
            ha='center', fontsize=34, fontweight='bold', color=WHITE)
    ax.text(0.5, 0.51, "(XAI — Explainable Artificial Intelligence)",
            ha='center', fontsize=18, color="#bfdbfe")
    ax.text(0.5, 0.38, "Sujet C.6  ·  IA Probabiliste, Théorie des Jeux et Machine Learning",
            ha='center', fontsize=12, color="#93c5fd")

    ax.text(0.5, 0.20, "MALAK El Idrissi & Joe Boueri", ha='center', fontsize=15,
            fontweight='bold', color=C_ACCENT)
    
    footer(fig, 1)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Contexte réglementaire
# ══════════════════════════════════════════════════════════════════════════════
def slide_contexte(pdf):
    fig = new_fig()
    header_bar(fig, "Contexte réglementaire", "Pourquoi l'IA Explicable est-elle obligatoire en finance ?")
    footer(fig, 2)

    ax = content_area(fig)

    boxes = [
        (0.01, "RGPD — Article 22", "#1e3a5f",
         "• Droit à l'explication pour toute\n  décision automatisée\n• Obligatoire depuis 2018\n• Amende jusqu'à 4% CA mondial"),
        (0.34, "EU AI Act (2024)", "#1e3a5f",
         "• Scoring crédit = système\n  haut risque (Annexe III)\n• Transparence obligatoire\n• Documentation complète requise"),
        (0.67, "Bâle III / IV", "#1e3a5f",
         "• Validation réglementaire\n  des modèles bancaires\n• Modèle interprétable de\n  référence obligatoire"),
    ]

    for xpos, title, color, text in boxes:
        rect = mpatches.FancyBboxPatch((xpos, 0.10), 0.30, 0.82,
                                        boxstyle="round,pad=0.02",
                                        linewidth=1.5, edgecolor=C_BLUE,
                                        facecolor=color)
        ax.add_patch(rect)
        ax.text(xpos + 0.15, 0.87, title, ha='center', fontsize=13,
                fontweight='bold', color=C_ACCENT)
        ax.text(xpos + 0.15, 0.50, text, ha='center', va='center',
                fontsize=11, color=WHITE, linespacing=1.6)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Dataset German Credit
# ══════════════════════════════════════════════════════════════════════════════
def slide_dataset(pdf):
    fig = new_fig()
    header_bar(fig, "Dataset — German Credit (UCI)", "Hofmann, Université de Hambourg")
    footer(fig, 3)

    ax = content_area(fig)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Statistiques
    stats = [
        ("1 000", "instances"),
        ("20", "features"),
        ("70 / 30", "Good / Bad"),
        ("0", "valeurs\nmanquantes"),
    ]
    for i, (val, lbl) in enumerate(stats):
        xc = 0.12 + i * 0.22
        rect = mpatches.FancyBboxPatch((xc - 0.09, 0.52), 0.18, 0.38,
                                        boxstyle="round,pad=0.02",
                                        linewidth=1, edgecolor=C_BLUE, facecolor="#1e3a5f")
        ax.add_patch(rect)
        ax.text(xc, 0.76, val, ha='center', fontsize=24, fontweight='bold', color=C_ACCENT)
        ax.text(xc, 0.60, lbl, ha='center', fontsize=10, color=WHITE)

    # Features sensibles
    ax.text(0.5, 0.44, "Features sensibles extraites :", ha='center', fontsize=12,
            fontweight='bold', color="#93c5fd")
    ax.text(0.5, 0.34, "gender  (dérivé de personal_status : male / female)",
            ha='center', fontsize=11, color=WHITE)
    ax.text(0.5, 0.25, "age_group  (young <25 · middle 25–39 · senior 40–59 · elderly 60+)",
            ha='center', fontsize=11, color=WHITE)

    # Top features
    ax.text(0.5, 0.14, "Top features : credit_amount · duration · age · checking_account · credit_history",
            ha='center', fontsize=11, color=C_GRAY, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Pipeline ML
# ══════════════════════════════════════════════════════════════════════════════
def slide_pipeline(pdf):
    fig = new_fig()
    header_bar(fig, "Pipeline Machine Learning", "De la donnée brute au dashboard explicable")
    footer(fig, 4)

    ax = content_area(fig)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    steps = [
        (0.05, "UCI\nRepository", C_BLUE),
        (0.22, "data_loader\n+ mapping", "#1e3a5f"),
        (0.39, "preprocessing\nLabelEnc · Scaler\n70/10/20", "#1e3a5f"),
        (0.56, "Modèles\nLR · XGB · LGBM", "#1e3a5f"),
        (0.73, "XAI\nSHAP · LIME · CF", "#1e3a5f"),
        (0.90, "Dashboard\nStreamlit", C_GREEN),
    ]

    for xc, label, color in steps:
        rect = mpatches.FancyBboxPatch((xc - 0.075, 0.38), 0.15, 0.38,
                                        boxstyle="round,pad=0.015",
                                        linewidth=1.5, edgecolor=C_ACCENT, facecolor=color)
        ax.add_patch(rect)
        ax.text(xc, 0.57, label, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color=WHITE, linespacing=1.5)

    # Flèches
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 0.075
        x2 = steps[i+1][0] - 0.075
        ax.annotate("", xy=(x2, 0.57), xytext=(x1, 0.57),
                    arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2))

    # Sous-branches XAI
    labels_xai = ["SHAP\nTreeExplainer", "LIME\nLocal", "Contrefactuels\nGradient"]
    xs_xai = [0.62, 0.73, 0.84]
    for x, lbl in zip(xs_xai, labels_xai):
        rect = mpatches.FancyBboxPatch((x - 0.055, 0.06), 0.11, 0.22,
                                        boxstyle="round,pad=0.01",
                                        linewidth=1, edgecolor="#6b7280", facecolor="#0f2237")
        ax.add_patch(rect)
        ax.text(x, 0.17, lbl, ha='center', va='center', fontsize=8.5,
                color="#93c5fd", linespacing=1.4)
        ax.annotate("", xy=(x, 0.28), xytext=(0.73, 0.38),
                    arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1, linestyle='dashed'))

    ax.text(0.5, 0.96, "Split stratifié · class_weight='balanced' · early stopping LightGBM",
            ha='center', fontsize=10, color=C_GRAY, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Performances des modèles
# ══════════════════════════════════════════════════════════════════════════════
def slide_performances(pdf):
    fig = new_fig()
    header_bar(fig, "Performances des modèles", "Évaluation sur le jeu de test (200 instances)")
    footer(fig, 5)

    ax_main = fig.add_axes([0.04, 0.055, 0.55, 0.755])
    ax_main.set_facecolor(C_DARK)

    models = ["Logistic\nRegression", "XGBoost", "LightGBM"]
    metrics = {
        "ROC-AUC":   [0.762, 0.824, 0.841],
        "Accuracy":  [0.735, 0.780, 0.795],
        "F1-Score":  [0.710, 0.770, 0.790],
    }
    colors_bars = [C_GRAY, C_BLUE, C_GREEN]
    x = np.arange(len(models))
    width = 0.25
    offsets = [-0.25, 0, 0.25]

    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax_main.bar(x + offsets[i], values, width,
                           label=metric, color=colors_bars[i], alpha=0.85, edgecolor='white', lw=0.5)
        for bar, v in zip(bars, values):
            ax_main.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f"{v:.3f}", ha='center', va='bottom', fontsize=7.5, color=WHITE)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(models, fontsize=10, color=WHITE)
    ax_main.set_ylim(0.60, 0.90)
    ax_main.set_ylabel("Score", color=WHITE, fontsize=10)
    ax_main.tick_params(colors=WHITE)
    ax_main.set_facecolor(C_DARK)
    ax_main.spines[:].set_color(C_GRAY)
    ax_main.legend(fontsize=9, facecolor="#1e3a5f", labelcolor=WHITE, loc='lower right')

    # Tableau récapitulatif
    ax_t = fig.add_axes([0.62, 0.055, 0.36, 0.755])
    ax_t.set_facecolor(C_DARK)
    ax_t.axis('off')
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

    ax_t.text(0.5, 0.95, "Récapitulatif", ha='center', fontsize=12,
              fontweight='bold', color=C_ACCENT)

    rows = [
        ("Modèle", "AUC", "F1", "Acc", True),
        ("Log. Reg.", "0.762", "0.71", "73.5%", False),
        ("XGBoost", "0.824", "0.77", "78.0%", False),
        ("LightGBM ★", "0.841", "0.79", "79.5%", False),
    ]
    ys = [0.82, 0.68, 0.52, 0.36]
    for (m, auc, f1, acc, is_header), y in zip(rows, ys):
        color = C_ACCENT if is_header else (C_GREEN if m.startswith("LightGBM") else WHITE)
        fw = 'bold' if is_header or m.startswith("LightGBM") else 'normal'
        ax_t.text(0.05, y, m,  fontsize=9.5, color=color, fontweight=fw)
        ax_t.text(0.52, y, auc, fontsize=9.5, color=color, fontweight=fw)
        ax_t.text(0.70, y, f1,  fontsize=9.5, color=color, fontweight=fw)
        ax_t.text(0.87, y, acc, fontsize=9.5, color=color, fontweight=fw)

    ax_t.axhline(y=0.75, xmin=0, xmax=1, color=C_GRAY, lw=0.8)
    ax_t.text(0.5, 0.18, "+7.9 pts AUC\nvs baseline",
              ha='center', fontsize=11, fontweight='bold', color=C_GREEN, linespacing=1.5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — SHAP Explication globale
# ══════════════════════════════════════════════════════════════════════════════
def slide_shap_global(pdf):
    fig = new_fig()
    header_bar(fig, "SHAP — Explication globale", "SHapley Additive exPlanations — Théorie des jeux coopératifs")
    footer(fig, 6)

    ax = fig.add_axes([0.04, 0.055, 0.50, 0.755])
    ax.set_facecolor(C_DARK)

    features = ["credit_amount", "duration", "age", "checking_account",
                "credit_history", "savings_account", "purpose",
                "employment_since", "installment_rate", "property"]
    importance = [0.245, 0.198, 0.156, 0.134, 0.112, 0.089, 0.067, 0.054, 0.043, 0.031]
    colors_shap = [C_BLUE if i < 5 else C_GRAY for i in range(len(features))]

    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance[::-1], color=colors_shap[::-1], edgecolor='white', lw=0.3, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features[::-1], fontsize=9, color=WHITE)
    ax.set_xlabel("Importance SHAP moyenne |φᵢ|", color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE)
    ax.set_facecolor(C_DARK)
    ax.spines[:].set_color(C_GRAY)

    for i, v in enumerate(importance[::-1]):
        ax.text(v + 0.003, i, f"{v:.3f}", va='center', fontsize=8, color=WHITE)

    # Texte théorique
    ax_t = fig.add_axes([0.57, 0.055, 0.41, 0.755])
    ax_t.set_facecolor(C_DARK)
    ax_t.axis('off')
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

    ax_t.text(0.5, 0.95, "Formule SHAP", ha='center', fontsize=12,
              fontweight='bold', color=C_ACCENT)
    ax_t.text(0.5, 0.84, "f(x) = φ₀ + Σ φᵢ(xᵢ)", ha='center', fontsize=14,
              color=C_BLUE, fontfamily='monospace')

    points = [
        "• φ₀ = prédiction moyenne (expected value)",
        "• φᵢ = contribution marginale de la feature i",
        "• Propriété d'efficience : Σφᵢ = f(x) − φ₀",
        "• Symétrie + factice + additivité garanties",
        "",
        "Implémentation :",
        "• TreeExplainer (exact, exploite les arbres)",
        "• Complexité O(TL²M) vs O(2ᴹ) naïf",
    ]
    for j, pt in enumerate(points):
        ax_t.text(0.04, 0.72 - j * 0.09, pt, fontsize=9.5, color=WHITE, linespacing=1.3)

    ax_t.text(0.5, 0.04, "credit_amount et duration = 44.3% de l'importance totale",
              ha='center', fontsize=9, color=C_GREEN, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — SHAP Explication locale
# ══════════════════════════════════════════════════════════════════════════════
def slide_shap_local(pdf):
    fig = new_fig()
    header_bar(fig, "SHAP — Explication locale", "Instance refusée — probabilité 32%")
    footer(fig, 7)

    ax = fig.add_axes([0.04, 0.12, 0.54, 0.68])
    ax.set_facecolor(C_DARK)

    contributions = {
        "credit_amount\n(élevé)": -0.18,
        "checking_account\n(no account)": -0.14,
        "credit_history\n(critical)": -0.11,
        "duration\n(36 mois)": -0.07,
        "savings_account\n(faible)": -0.05,
        "age\n(28 ans)": +0.06,
        "employment_since\n(3 ans)": +0.03,
    }

    feats = list(contributions.keys())
    vals  = list(contributions.values())
    colors_cf = [C_RED if v < 0 else C_GREEN for v in vals]

    y_pos = np.arange(len(feats))
    ax.barh(y_pos, vals, color=colors_cf, edgecolor='white', lw=0.3, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=8.5, color=WHITE)
    ax.axvline(x=0, color=WHITE, lw=1)
    ax.set_xlabel("Contribution SHAP (φᵢ)", color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE)
    ax.set_facecolor(C_DARK)
    ax.spines[:].set_color(C_GRAY)

    for i, v in enumerate(vals):
        xpos = v + 0.003 if v >= 0 else v - 0.003
        ha = 'left' if v >= 0 else 'right'
        ax.text(xpos, i, f"{v:+.2f}", va='center', fontsize=8, color=WHITE, ha=ha)

    # Jauge de probabilité
    ax_g = fig.add_axes([0.62, 0.55, 0.36, 0.24])
    ax_g.set_facecolor(C_DARK)
    ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
    ax_g.axis('off')
    ax_g.text(0.5, 0.92, "Probabilité d'approbation", ha='center', fontsize=10,
              fontweight='bold', color=C_ACCENT)
    prob = 0.32
    rect_bg = mpatches.FancyBboxPatch((0.05, 0.35), 0.90, 0.30,
                                       boxstyle="round,pad=0.01",
                                       facecolor="#1e3a5f", linewidth=0)
    ax_g.add_patch(rect_bg)
    rect_fill = mpatches.FancyBboxPatch((0.05, 0.35), 0.90 * prob, 0.30,
                                         boxstyle="round,pad=0.01",
                                         facecolor=C_RED, linewidth=0, alpha=0.85)
    ax_g.add_patch(rect_fill)
    ax_g.text(0.5, 0.50, f"{prob*100:.0f}%", ha='center', va='center',
              fontsize=18, fontweight='bold', color=WHITE)
    ax_g.text(0.5, 0.12, "REFUSÉ", ha='center', fontsize=14,
              fontweight='bold', color=C_RED)

    # Explication textuelle
    ax_e = fig.add_axes([0.62, 0.10, 0.36, 0.41])
    ax_e.set_facecolor("#0f2237")
    ax_e.set_xlim(0, 1); ax_e.set_ylim(0, 1)
    ax_e.axis('off')
    ax_e.text(0.5, 0.93, "Interprétation", ha='center', fontsize=10,
              fontweight='bold', color=C_ACCENT)
    lines = [
        "Facteurs défavorables :",
        "  → Montant du crédit trop élevé (−0.18)",
        "  → Absence de compte courant (−0.14)",
        "  → Historique critique (−0.11)",
        "",
        "Facteur favorable :",
        "  → Âge (28 ans) : +0.06",
    ]
    for j, line in enumerate(lines):
        color = C_RED if "défavorables" in line else (C_GREEN if "favorable" in line else WHITE)
        ax_e.text(0.05, 0.80 - j * 0.12, line, fontsize=8.5, color=color)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — LIME
# ══════════════════════════════════════════════════════════════════════════════
def slide_lime(pdf):
    fig = new_fig()
    header_bar(fig, "LIME — Explication locale", "Local Interpretable Model-agnostic Explanations")
    footer(fig, 8)

    # Graphique LIME (contributions locales)
    ax = fig.add_axes([0.04, 0.12, 0.44, 0.68])
    ax.set_facecolor(C_DARK)

    feats_lime = ["credit_amount > 4800", "checking_account=A11",
                  "duration > 24", "credit_history=A32",
                  "savings_account=A61", "age <= 25", "purpose=A43"]
    vals_lime  = [-0.162, -0.121, -0.098, -0.076, -0.045, +0.052, +0.028]
    colors_lime = [C_RED if v < 0 else C_GREEN for v in vals_lime]

    y_pos = np.arange(len(feats_lime))
    ax.barh(y_pos, vals_lime, color=colors_lime, edgecolor='white', lw=0.3, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats_lime, fontsize=8.5, color=WHITE)
    ax.axvline(x=0, color=WHITE, lw=1)
    ax.set_xlabel("Contribution LIME locale", color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE)
    ax.set_facecolor(C_DARK)
    ax.spines[:].set_color(C_GRAY)

    # Comparaison SHAP vs LIME
    ax_t = fig.add_axes([0.52, 0.12, 0.46, 0.68])
    ax_t.set_facecolor(C_DARK)
    ax_t.axis('off')
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

    ax_t.text(0.5, 0.96, "SHAP vs LIME — Comparaison", ha='center', fontsize=12,
              fontweight='bold', color=C_ACCENT)

    headers = ["Critère", "SHAP", "LIME"]
    rows = [
        ["Fondement",        "Théorie des jeux", "Approx. linéaire"],
        ["Cohérence globale","Oui",               "Non"],
        ["Stabilité",        "Élevée",            "Variable"],
        ["Portée",           "Globale + locale",  "Locale uniquement"],
        ["Modèle-agnostique","Non (TreeExp.)",    "Oui"],
    ]

    ys_h = 0.83
    xs = [0.02, 0.38, 0.70]
    for j, h in enumerate(headers):
        ax_t.text(xs[j], ys_h, h, fontsize=10, fontweight='bold', color=C_ACCENT)
    ax_t.axhline(y=0.78, xmin=0, xmax=1, color=C_GRAY, lw=0.8)

    for i, row in enumerate(rows):
        y_row = 0.68 - i * 0.13
        for j, cell in enumerate(row):
            color = WHITE
            if j > 0:
                if cell in ("Oui", "Élevée", "Globale + locale"):
                    color = C_GREEN
                elif cell in ("Non", "Variable", "Locale uniquement", "Non (TreeExp.)"):
                    color = "#f87171"
            ax_t.text(xs[j], y_row, cell, fontsize=9, color=color)

    ax_t.text(0.5, 0.07, "Concordance top-5 features : ~78% des instances",
              ha='center', fontsize=10, color=C_GREEN, fontweight='bold', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Contrefactuels
# ══════════════════════════════════════════════════════════════════════════════
def slide_contrefactuels(pdf):
    fig = new_fig()
    header_bar(fig, "Explications contrefactuelles", "\"Que faudrait-il modifier pour être accepté ?\"")
    footer(fig, 9)

    ax = content_area(fig)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Instance originale → Contrefactuel
    ax.text(0.5, 0.92, "Exemple type — Instance refusée (probabilité 31%)", ha='center',
            fontsize=11, fontweight='bold', color=C_ACCENT)

    headers_cf = ["Feature", "Valeur originale", "Valeur suggérée", "Changement"]
    rows_cf = [
        ("credit_amount",    "8 500 DM",    "4 200 DM",   "−4 300 DM"),
        ("duration",         "36 mois",     "24 mois",    "−12 mois"),
        ("checking_account", "no account",  "0–200 DM",   "ouvrir compte"),
    ]

    xs_cf = [0.04, 0.28, 0.52, 0.75]
    y_start = 0.80

    for j, h in enumerate(headers_cf):
        rect = mpatches.FancyBboxPatch((xs_cf[j] - 0.01, y_start - 0.03), 0.24, 0.08,
                                        boxstyle="square,pad=0.01",
                                        linewidth=0, facecolor=C_BLUE)
        ax.add_patch(rect)
        ax.text(xs_cf[j] + 0.11, y_start + 0.01, h, ha='center', fontsize=10,
                fontweight='bold', color=WHITE)

    for i, (feat, orig, sugg, chg) in enumerate(rows_cf):
        y_row = 0.66 - i * 0.14
        for j, val in enumerate([feat, orig, sugg, chg]):
            color = C_RED if j == 1 else (C_GREEN if j == 2 else (C_ACCENT if j == 3 else WHITE))
            ax.text(xs_cf[j] + 0.11, y_row, val, ha='center', fontsize=10, color=color)

    # Résultat
    ax.text(0.5, 0.21, "Probabilité après modifications : 67%  →  DÉCISION INVERSÉE ✓",
            ha='center', fontsize=13, fontweight='bold', color=C_GREEN)

    # Algorithme
    ax.text(0.5, 0.12, "Algorithme : descente de gradient (diff. finies) + recherche discrète catégorielles",
            ha='center', fontsize=9.5, color=C_GRAY, style='italic')
    ax.text(0.5, 0.04, "Taux de succès : ~82% des instances refusées → contrefactuel valide trouvé",
            ha='center', fontsize=9.5, color=C_GRAY, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Fairness genre
# ══════════════════════════════════════════════════════════════════════════════
def slide_fairness_genre(pdf):
    fig = new_fig()
    header_bar(fig, "Audit de Fairness — Genre", "Fairlearn · Parité démographique + Equalized Odds")
    footer(fig, 10)

    ax = fig.add_axes([0.04, 0.12, 0.44, 0.68])
    ax.set_facecolor(C_DARK)

    groups  = ["Femmes", "Hommes"]
    tpr     = [0.76, 0.79]
    fpr     = [0.18, 0.21]

    x = np.arange(len(groups))
    w = 0.35
    b1 = ax.bar(x - w/2, tpr, w, label="TPR (recall)", color=C_GREEN, alpha=0.85)
    b2 = ax.bar(x + w/2, fpr, w, label="FPR", color=C_RED, alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha='center', fontsize=9, color=WHITE)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=WHITE, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Taux", color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE)
    ax.set_facecolor(C_DARK)
    ax.spines[:].set_color(C_GRAY)
    ax.legend(fontsize=9, facecolor="#1e3a5f", labelcolor=WHITE)
    ax.set_title("TPR / FPR par genre", color=WHITE, fontsize=10)

    # Métriques fairness
    ax_m = fig.add_axes([0.52, 0.12, 0.46, 0.68])
    ax_m.set_facecolor(C_DARK)
    ax_m.axis('off')
    ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1)

    ax_m.text(0.5, 0.96, "Métriques de fairness — Genre", ha='center', fontsize=11,
              fontweight='bold', color=C_ACCENT)

    metrics_f = [
        ("Parité démographique", "diff.", "~0.062", "Bon ✓", C_GREEN),
        ("Equalized odds",       "diff.", "~0.033", "Excellent ✓", C_GREEN),
    ]
    seuils = [
        "< 0.05  → Excellent",
        "0.05–0.10  → Bon",
        "0.10–0.20  → Moyen",
        "> 0.20  → Insuffisant",
    ]

    for i, (name, kind, val, interp, color) in enumerate(metrics_f):
        y_row = 0.80 - i * 0.18
        rect = mpatches.FancyBboxPatch((0.02, y_row - 0.06), 0.96, 0.14,
                                        boxstyle="round,pad=0.01",
                                        linewidth=1, edgecolor=color, facecolor="#0f2237")
        ax_m.add_patch(rect)
        ax_m.text(0.06, y_row + 0.02, f"{name} ({kind}) = {val}", fontsize=10, color=WHITE)
        ax_m.text(0.94, y_row + 0.02, interp, ha='right', fontsize=10, fontweight='bold', color=color)

    ax_m.text(0.5, 0.43, "Grille d'interprétation :", ha='center', fontsize=10,
              fontweight='bold', color=C_GRAY)
    for i, s in enumerate(seuils):
        ax_m.text(0.5, 0.34 - i * 0.09, s, ha='center', fontsize=9, color=C_GRAY)

    ax_m.text(0.5, 0.04, "Le modèle est équitable par genre ✓",
              ha='center', fontsize=11, fontweight='bold', color=C_GREEN)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Fairness âge + mitigation
# ══════════════════════════════════════════════════════════════════════════════
def slide_fairness_age(pdf):
    fig = new_fig()
    header_bar(fig, "Audit de Fairness — Âge & Mitigation", "ExponentiatedGradient — trade-off équité/performance")
    footer(fig, 11)

    ax = fig.add_axes([0.04, 0.12, 0.44, 0.68])
    ax.set_facecolor(C_DARK)

    groups_age = ["young\n(<25)", "middle\n(25–39)", "senior\n(40–59)", "elderly\n(60+)"]
    approbation = [0.58, 0.72, 0.75, 0.68]
    colors_age = [C_RED, C_GREEN, C_GREEN, C_BLUE]

    bars = ax.bar(groups_age, approbation, color=colors_age, edgecolor='white', lw=0.5, width=0.5)
    for bar, v in zip(bars, approbation):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.0%}", ha='center', fontsize=10, fontweight='bold', color=WHITE)
    ax.set_ylim(0, 0.95)
    ax.axhline(y=0.71, color=C_ACCENT, lw=1.5, linestyle='--', label="Moy. globale ~71%")
    ax.set_ylabel("Taux d'approbation", color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE)
    ax.set_facecolor(C_DARK)
    ax.spines[:].set_color(C_GRAY)
    ax.legend(fontsize=9, facecolor="#1e3a5f", labelcolor=WHITE)
    ax.set_title("Taux d'approbation par tranche d'âge", color=WHITE, fontsize=9)

    # Tableau mitigation
    ax_t = fig.add_axes([0.52, 0.12, 0.46, 0.68])
    ax_t.set_facecolor(C_DARK)
    ax_t.axis('off')
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

    ax_t.text(0.5, 0.96, "Mitigation — ExponentiatedGradient", ha='center', fontsize=10,
              fontweight='bold', color=C_ACCENT)

    headers_m = ["Métrique", "Avant", "Après", "Δ"]
    rows_m = [
        ("Accuracy",        "78.0%", "74.2%", "−3.8 pts", C_RED),
        ("DP diff. (âge)",  "0.136", "0.068", "−0.068",   C_GREEN),
        ("EO diff. (âge)",  "0.100", "0.051", "−0.049",   C_GREEN),
    ]
    xs_m = [0.02, 0.34, 0.56, 0.76]
    for j, h in enumerate(headers_m):
        ax_t.text(xs_m[j], 0.84, h, fontsize=9.5, fontweight='bold', color=C_ACCENT)
    ax_t.axhline(y=0.79, xmin=0, xmax=1, color=C_GRAY, lw=0.7)

    for i, (name, avant, apres, delta, color) in enumerate(rows_m):
        y_r = 0.68 - i * 0.15
        ax_t.text(xs_m[0], y_r, name,  fontsize=9, color=WHITE)
        ax_t.text(xs_m[1], y_r, avant, fontsize=9, color=WHITE)
        ax_t.text(xs_m[2], y_r, apres, fontsize=9, color=WHITE)
        ax_t.text(xs_m[3], y_r, delta, fontsize=9, fontweight='bold', color=color)

    ax_t.text(0.5, 0.20, "Trade-off typique :", ha='center', fontsize=10,
              fontweight='bold', color=C_GRAY)
    ax_t.text(0.5, 0.10, "−3.8 pts accuracy pour −50% de biais d'âge",
              ha='center', fontsize=10, color=WHITE)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Dashboard démo
# ══════════════════════════════════════════════════════════════════════════════
def slide_dashboard(pdf):
    fig = new_fig()
    header_bar(fig, "Dashboard Streamlit — 5 pages", "Démonstration interactive")
    footer(fig, 12)

    ax = content_area(fig)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    pages = [
        ("[1] Accueil",      "#1e3a5f", 0.01,
         "• Contexte réglementaire\n• Stats dataset\n• Distribution cible"),
        ("[2] Prédiction",   "#1e3a5f", 0.21,
         "• Sélection test ou saisie manuelle\n• 3 modèles disponibles\n• Jauge de confiance"),
        ("[3] Explicabilité","#1e3a5f", 0.41,
         "• SHAP global + local\n• LIME local\n• Contrefactuels"),
        ("[4] Fairness",     "#1e3a5f", 0.61,
         "• Parité démographique\n• Equalized odds\n• Métriques par groupe"),
        ("[5] Comparaison",  "#1e3a5f", 0.81,
         "• Tableau multi-modèles\n• Radar chart\n• Graphiques groupés"),
    ]

    for title, color, x, desc in pages:
        rect = mpatches.FancyBboxPatch((x, 0.10), 0.18, 0.82,
                                        boxstyle="round,pad=0.02",
                                        linewidth=1.5, edgecolor=C_BLUE, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.09, 0.86, title, ha='center', fontsize=10,
                fontweight='bold', color=C_ACCENT)
        ax.text(x + 0.09, 0.54, desc, ha='center', va='center',
                fontsize=8.5, color=WHITE, linespacing=1.6)

    ax.text(0.5, 0.04, "streamlit run src/dashboard/app.py  →  http://localhost:8501",
            ha='center', fontsize=10, color=C_GRAY, fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Architecture technique
# ══════════════════════════════════════════════════════════════════════════════
def slide_architecture(pdf):
    fig = new_fig()
    header_bar(fig, "Architecture technique", "Stack Python — modules et dépendances")
    footer(fig, 13)

    ax = content_area(fig)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    modules = [
        ("src/data_loader.py",         0.02, 0.70, "#1a3a5f", "Téléchargement UCI\n+ mapping catégoriel"),
        ("src/preprocessing.py",        0.02, 0.44, "#1a3a5f", "LabelEncoder\nStandardScaler · split"),
        ("src/models/lightgbm_model.py",0.38, 0.70, "#0f2237", "LightGBM\nROC-AUC 0.841"),
        ("src/models/xgboost_model.py", 0.38, 0.44, "#0f2237", "XGBoost\nROC-AUC 0.824"),
        ("src/explainability/shap_explainer.py", 0.62, 0.70, "#0f2237", "SHAP\nTreeExplainer"),
        ("src/explainability/lime_explainer.py", 0.62, 0.44, "#0f2237", "LIME\nTabularExplainer"),
        ("src/explainability/counterfactual.py", 0.62, 0.18, "#0f2237", "Contrefactuels\nGradient-based"),
        ("src/fairness/fairness_audit.py",        0.38, 0.18, "#1a3a5f", "Fairlearn\nMetricFrame"),
        ("src/dashboard/app.py",                   0.02, 0.18, "#1a3a5f", "Streamlit\n5 pages"),
    ]

    for name, x, y, color, desc in modules:
        short = name.split("/")[-1].replace(".py", "")
        rect = mpatches.FancyBboxPatch((x, y - 0.05), 0.22, 0.18,
                                        boxstyle="round,pad=0.015",
                                        linewidth=1, edgecolor=C_BLUE, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + 0.11, y + 0.09, short, ha='center', fontsize=8.5,
                fontweight='bold', color=C_ACCENT)
        ax.text(x + 0.11, y + 0.015, desc, ha='center', va='center',
                fontsize=7.5, color=WHITE, linespacing=1.4)

    ax.text(0.5, 0.04,
            "Python 3.11 · XGBoost · LightGBM · SHAP · LIME · Fairlearn · Streamlit · scikit-learn · joblib",
            ha='center', fontsize=9, color=C_GRAY, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Limites & Perspectives
# ══════════════════════════════════════════════════════════════════════════════
def slide_perspectives(pdf):
    fig = new_fig()
    header_bar(fig, "Limites & Perspectives", "Axes d'amélioration prioritaires")
    footer(fig, 14)

    ax = content_area(fig)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Limites
    ax.text(0.24, 0.93, "Limites actuelles", ha='center', fontsize=12,
            fontweight='bold', color=C_RED)
    limites = [
        "Dataset 1 000 instances (années 1990)",
        "Pas de validation croisée (split unique)",
        "Probabilités non calibrées",
        "LIME instable entre runs",
        "Fairness : pas d'intersectionnalité",
    ]
    for i, lim in enumerate(limites):
        ax.text(0.02, 0.80 - i * 0.12, f"⚠  {lim}", fontsize=10, color="#f87171")

    # Perspectives
    ax.text(0.74, 0.93, "Améliorations prioritaires", ha='center', fontsize=12,
            fontweight='bold', color=C_GREEN)
    improv = [
        ("Calibration Platt scaling",     "Élevé",  "Faible"),
        ("Validation croisée k=5",         "Élevé",  "Moyen"),
        ("SMOTE déséquilibre",             "Moyen",  "Faible"),
        ("DiCE contrefactuels diversifiés","Élevé",  "Moyen"),
        ("API REST FastAPI",               "Élevé",  "Moyen"),
    ]
    for i, (name, impact, compl) in enumerate(improv):
        y_r = 0.80 - i * 0.12
        ax.text(0.51, y_r, f"→  {name}", fontsize=9.5, color=WHITE)
        color_imp = C_GREEN if impact == "Élevé" else C_ACCENT
        ax.text(0.90, y_r, impact, fontsize=9, color=color_imp, ha='right', fontweight='bold')

    # Extensions
    ax.axhline(y=0.18, xmin=0, xmax=1, color=C_GRAY, lw=0.8)
    ax.text(0.5, 0.11, "Extensions : Lending Club (880k prêts)  ·  Monitoring data drift (Evidently)  ·  Conformité RGPD automatisée",
            ha='center', fontsize=10, color=C_GRAY, style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Conclusion
# ══════════════════════════════════════════════════════════════════════════════
def slide_conclusion(pdf):
    fig = new_fig()
    header_bar(fig, "Conclusion", "Credit Scoring avec IA Explicable — Bilan")
    footer(fig, 15)

    ax = content_area(fig)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    achievements = [
        ("Modèle performant",     "LightGBM — ROC-AUC 0.841  (+7.9 pts vs baseline)"),
        ("Explicabilité globale", "SHAP TreeExplainer — top 10 features identifiées"),
        ("Explicabilité locale",  "SHAP + LIME — concordance ~78% · textes dynamiques"),
        ("Contrefactuels",        "Taux de succès ~82% · réponse réglementaire RGPD Art.22"),
        ("Audit fairness",        "Genre : excellent · Âge : moyen · mitigation −50% biais"),
        ("Dashboard interactif",  "Streamlit 5 pages · déploiement local < 5 secondes"),
        ("Conformité réglementaire","RGPD Art.22 · EU AI Act haut risque · Bâle III/IV"),
    ]

    for i, (title, detail) in enumerate(achievements):
        y_row = 0.87 - i * 0.115
        # Icône checkmark
        circle = plt.Circle((0.025, y_row + 0.01), 0.017, color=C_GREEN, transform=ax.transData)
        ax.add_patch(circle)
        ax.text(0.025, y_row + 0.01, "✓", ha='center', va='center',
                fontsize=8, color=WHITE, fontweight='bold')
        ax.text(0.06, y_row + 0.02, title, fontsize=10.5, fontweight='bold', color=C_ACCENT)
        ax.text(0.06, y_row - 0.03, detail, fontsize=9.5, color=WHITE)

    

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with PdfPages(OUT_PATH) as pdf:
        slide_titre(pdf)
        slide_contexte(pdf)
        slide_dataset(pdf)
        slide_pipeline(pdf)
        slide_performances(pdf)
        slide_shap_global(pdf)
        slide_shap_local(pdf)
        slide_lime(pdf)
        slide_contrefactuels(pdf)
        slide_fairness_genre(pdf)
        slide_fairness_age(pdf)
        slide_dashboard(pdf)
        slide_architecture(pdf)
        slide_perspectives(pdf)
        slide_conclusion(pdf)

        d = pdf.infodict()
        d['Title']   = 'Credit Scoring avec IA Explicable (XAI)'
        d['Author']  = 'MALAK El Idrissi et Joe Boueri — ECE Paris Ing4 Finance Groupe 03'
        d['Subject'] = ' Sujet C.6'

    print(f"PDF généré : {OUT_PATH}")
    print(f"15 slides — {os.path.getsize(OUT_PATH) // 1024} Ko")


if __name__ == "__main__":
    main()
