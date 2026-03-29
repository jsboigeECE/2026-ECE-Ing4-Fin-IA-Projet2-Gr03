# 15 — Visualization Masterplan

---

## Governing Principle

Every chart must earn its place. Each chart has exactly one message. Charts that could be removed without losing a key argument must be removed. There are no decorative figures in this project.

**Total figures target:** 8–10 figures for the full report. 5–6 for the presentation. Zero dashboard widgets.

---

## Figure Catalog (Ranked by Impact)

---

### FIGURE 1 — Rolling Coverage Plot (HERO VISUAL)
**Rank:** 1 — Non-negotiable  
**Message:** ACI and EnbPI maintain coverage better than static CP and GARCH during the COVID crisis and 2022 bear market  
**Type:** Line chart (time series)  
**X-axis:** Date (2020–2024)  
**Y-axis:** 60-day rolling empirical coverage rate (0 to 1)  
**Series:** Split CP (red), ACI (green), EnbPI (blue), GARCH (orange), HistSim (grey)  
**Annotations:** Vertical shaded bands for COVID crash (Feb–Apr 2020) and 2022 bear (Jan–Dec 2022). Horizontal dashed line at nominal level (e.g., 0.95). VIX line (secondary right axis, light grey)  
**Expected insight:** Red line drops sharply below 0.95 dashed line during shaded bands. Green line stays closer to 0.95, with faster recovery.  
**Slide destination:** Slide 4 (Core Results)  
**File:** `figures/fig01_rolling_coverage.png`

---

### FIGURE 2 — Interval Width Over Time (COVID Zoom)
**Rank:** 2 — Non-negotiable  
**Message:** ACI and EnbPI dynamically widen during stress; Static CP and GARCH stay flat or spike chaotically  
**Type:** Line chart (time series), zoomed to Feb–Jun 2020  
**X-axis:** Date (2020-01-01 to 2020-06-30)  
**Y-axis:** Interval width (in return units, e.g., percentage points)  
**Series:** All 7 methods  
**Annotations:** SPY price (or return) on secondary axis. Key event dates labeled (Feb 19 all-time high, Mar 23 trough)  
**Expected insight:** ACI intervals begin widening 5–10 days before the worst market days (as early exceptions trigger α_t update). Static CP stays flat. GARCH spikes overnight.  
**Slide destination:** Slide 4 (Core Results) or Slide 5 (Stress Analysis)  
**File:** `figures/fig02_widths_covid_zoom.png`

---

### FIGURE 3 — Regime Coverage Heatmap
**Rank:** 3 — Very high value  
**Message:** All methods degrade under stress; adaptive methods degrade less  
**Type:** Heatmap table (methods × regimes)  
**Rows:** 7 methods  
**Columns:** 4 regime labels (calm, elevated, stress, crisis) + Full period  
**Cell values:** Empirical coverage rate at 95% nominal level  
**Color scale:** Green (≥ 0.95) → Yellow (0.85–0.95) → Red (< 0.85)  
**Expected insight:** ACI row stays greener across crisis column vs Static CP row which goes red  
**Slide destination:** Slide 5 (Regime Analysis)  
**File:** `figures/fig03_regime_coverage_heatmap.png`

---

### FIGURE 4 — Kupiec Test P-Value Comparison
**Rank:** 4 — High value (finance credibility)  
**Message:** Parametric models fail regulatory backtests during stress; conformal methods are more robust  
**Type:** Grouped bar chart  
**X-axis:** Methods  
**Y-axis:** Kupiec test p-value (log scale)  
**Color:** Period (full period = solid; COVID period = hatched)  
**Annotation:** Horizontal red dashed line at p = 0.05 (rejection threshold)  
**Expected insight:** GARCH and HistSim bar drops below red line during COVID. ACI bar stays above it (or is less severe).  
**Slide destination:** Slide 5 or supplementary  
**File:** `figures/fig04_kupiec_pvalues.png`

---

### FIGURE 5 — ACI Alpha_t Time Series
**Rank:** 5 — High theoretical value  
**Message:** ACI's self-correcting mechanism is visible and behaves as theorized  
**Type:** Dual-axis time series  
**Primary Y-axis (left):** α_t time series (effective miscoverage target)  
**Secondary Y-axis (right):** VIX  
**X-axis:** Date (full test period 2020–2024)  
**Expected insight:** α_t drops sharply during COVID (ACI forces wider intervals to recover coverage), then slowly rises as the market normalizes  
**Slide destination:** Slide 3 (Methods) or Slide 4 (Results intro)  
**File:** `figures/fig05_aci_alpha_dynamics.png`

---

### FIGURE 6 — Equity Curves: Decision Layer
**Rank:** 6 — Essential for finance application  
**Message:** Uncertainty-aware position sizing (ACI) reduces drawdown during crisis compared to buy-and-hold and static CP sizing  
**Type:** Equity curve (cumulative log-return, rebased to 100)  
**Series:** Buy-and-Hold (grey), Static CP Sizing (red), ACI Sizing (green), GARCH Vol Target (orange)  
**X-axis:** Date (2020–2024)  
**Y-axis:** Cumulative portfolio value (rebased to 100 at start)  
**Annotations:** Shaded bands for stress periods. Max drawdown annotated with arrows for BH and ACI Sizing.  
**Expected insight:** ACI Sizing equity curve has shallower drawdown in 2020 and 2022; converges back to BH performance in recovery (cost of protection is modest).  
**Slide destination:** Slide 6 (Finance Application)  
**File:** `figures/fig06_equity_curves.png`

---

### FIGURE 7 — Coverage vs Width Scatter (Efficiency Frontier)
**Rank:** 7 — High methodological value  
**Message:** Methods face a coverage-width tradeoff; ACI and CQR are on the Pareto frontier  
**Type:** Scatter plot  
**X-axis:** Mean interval width (efficiency — lower is better)  
**Y-axis:** Empirical coverage rate (coverage — higher is better)  
**One point per method** (7 points), labeled  
**Target zone:** Highlighted box at [correct coverage ± 0.02, width ≤ threshold]  
**Expected insight:** ACI and CQR are closest to the ideal corner (tight + well-covered). GARCH is narrow but under-covered (crisis). HistSim is wide and still under-covered.  
**Slide destination:** Slide 4 or supplementary  
**File:** `figures/fig07_coverage_width_scatter.png`

---

### FIGURE 8 — Exception Calendar Heatmap
**Rank:** 8 — High visual impact  
**Message:** Static CP and GARCH exceptions cluster in crisis months; ACI exceptions are more dispersed  
**Type:** Calendar heatmap (months × methods or months × years)  
**Cell value:** Exception count per month per method  
**Color:** White (0 exceptions) → Red (many exceptions)  
**Expected insight:** GARCH and HistSim show dark red in March 2020. ACI shows lighter coverage.  
**Slide destination:** Supplementary or Slide 5  
**File:** `figures/fig08_exception_calendar.png`

---

### FIGURE 9 — Position Sizing Time Series (COVID Zoom)
**Rank:** 9 — Supports decision layer narrative  
**Message:** ACI-based position sizing reacts to uncertainty; Static CP does not  
**Type:** Line chart  
**X-axis:** Date (Feb–Jun 2020)  
**Y-axis:** Position size as fraction of base (0 to 1.5)  
**Series:** ACI Sizing (green), Static CP Sizing (red), GARCH Vol Target (orange), Buy-and-Hold (flat grey line at 1.0)  
**Annotations:** Key event dates. VIX overlay on secondary axis.  
**Expected insight:** ACI sizing drops sharply during Feb-Mar 2020 crash. Static CP position sizing stays at cap.  
**Slide destination:** Slide 6 (Finance Application)  
**File:** `figures/fig09_position_sizing_covid.png`

---

### FIGURE 10 — Winkler Score Table (Optional)
**Rank:** 10 — Only include if time permits  
**Message:** A proper scoring rule unified assessment of all methods  
**Type:** Bar chart with error bars (bootstrapped 95% CI)  
**Lower is better for Winkler score**  
**Expected insight:** ACI has the best (lowest) Winkler score during stress periods  
**Slide destination:** Supplementary only  
**File:** `figures/fig10_winkler_scores.png`

---

## Figures Explicitly Rejected

| Figure Idea | Reason Rejected |
|---|---|
| Feature correlation matrix | No new scientific insight; filler chart |
| Ridge coefficient bar chart | Adds model explanation but not coverage story |
| QQ plots of returns | Motivational only; not a results chart |
| 3D surface plots | Impressive visually, impossible to read |
| Interactive plotly dashboard | Out of scope; no grading gain |
| Confusion matrix (for direction) | We are not doing classification |
| Training loss curves | No neural networks in project |
| Multiple rolling windows comparison | Too much noise; 60-day window is pre-specified |
| Pair plot of features vs target | Explanatory only; wastes slide space |

---

## Figure Production Standards

1. **Format:** All figures saved as high-resolution PNG (300 DPI) for print, plus SVG for slides
2. **Color palette:** Use colorblind-safe palette (ColorBrewer: Set1 for categorical, Blues/Reds for sequential)
3. **Font size:** Minimum 12pt labels for all axes; 10pt for legends. Legible when projected.
4. **No 3D charts.** Ever.
5. **Consistent color assignment across all figures:** ACI = always green, Static CP = always red, GARCH = always orange, HistSim = always grey, EnbPI = always blue, CQR = always purple
6. **All figures reproducible** from `src/visualization.py` with a single function call per figure
7. **Figures numbered 01–10** matching this masterplan
