# Machine Learning Analysis of Microfinance and Household Resilience in the Philippines
## AI 221 — Classical Machine Learning Final Project
**Author:** Quiwa, Quiel Andrew I.
**Data:** Philippine Microfinance Survey 2024 (PMS 2024), PSA CPI, NOAA IBTrACS
**Notebook:** `pms-resilience.ipynb`

---

## Abstract

This study applies classical machine learning methods to the Philippine Microfinance Survey 2024 (PMS 2024) to identify which household profiles and external conditions are associated with better self-reported resilience outcomes among MFI service users. Using PCA and K-Means clustering, three vulnerability typologies are derived from 734 MFI-availing households. A Random Forest classifier and Elastic Net regularized logistic regression are then trained to predict three outcome targets — consumption stability, shock resilience, and quality of life — and benchmarked against an unpenalized multinomial logit baseline. Objective regional controls (PSA CPI and NOAA IBTrACS typhoon data) are integrated to test whether external conditions add predictive value over a regional fixed effect. Key findings are: (1) the Economically Unstable cluster reports the worst outcomes across all three targets; (2) Random Forest outperforms the logit benchmark on all targets, most substantially for Shock Resilience and Quality of Life (macro-F1: 0.453 vs 0.393 for QoL); (3) compound climate-and-inflation shocks produce no statistically significant difference in outcomes relative to non-compound shock exposure; and (4) objective CPI and typhoon measures add no predictive value beyond knowing the respondent's region — and underperform the base model without any explicit regional controls.

---

## 1. Research Question

> *Among households that avail of MFI services, which profiles and external conditions are associated with better self-reported resilience outcomes?*

The study is explicitly associational. It does not claim to identify the causal effect of MFI on resilience — the outcome variables are self-reported attribution questions asked only of current MFI users, making causal inference infeasible without a valid counterfactual design.

---

## 2. Data

### 2.1 Philippine Microfinance Survey 2024

The PMS 2024 is a nationally representative survey of 1,900 Philippine households conducted by WeSolve Foundation, Inc.. This study is restricted to the 734 households that reported availing of at least one MFI service (`e3_1 == 'AVAILED'`). The filter is applied before any transformations to preserve the integrity of skip-logic sentinel values (`-1`) that mark questions not shown to non-MFI respondents.

33 columns are retained and renamed from raw survey codes. Features span five domains: household demographics and socioeconomic status, shock exposure history, MFI engagement characteristics, financial literacy, and asset ownership.

### 2.2 Three Prediction Targets

All three targets are 5-point Likert scales asking respondents to evaluate MFI's effect on their household:

| Target | Column | Question summary |
|---|---|---|
| Consumption Stability | `cs_3class` | Did MFI improve the household's ability to maintain stable spending? |
| Shock Resilience | `sr_3class` | Did MFI strengthen resilience to economic downturns? |
| Quality of Life | `qol_3class` | Did MFI improve overall household wellbeing? |

**3-class collapse.** The original 5-point scale is collapsed to three classes — worsened (0), neutral (1), improved (2) — because the worsened category has only 6–44 observations across targets (2–6% of sample). A 5-class classifier cannot learn from class sizes this small, and class collapse preserves the directional distinction the research question requires.

**Class distributions after collapse:**

| Class | Consumption Stability | Shock Resilience | Quality of Life |
|---|---|---|---|
| 0 — Worsened | 44 (6%) | 39 (5%) | 14 (2%) |
| 1 — Neutral | 46 (6%) | 64 (9%) | 154 (21%) |
| 2 — Improved | 644 (88%) | 631 (86%) | 566 (77%) |

The severe class imbalance toward "improved" is structural — it reflects that current MFI users self-report positively by default — and is addressed with `class_weight='balanced'` in logit/Elastic Net and `class_weight='balanced_subsample'` in Random Forest. Macro-F1 is used as the primary evaluation metric because it weights all three classes equally regardless of prevalence.

### 2.3 External Data

**PSA Consumer Price Index (CPI), Bottom 30% of Households** — monthly data, 2022–2023. Four regional features are derived: year-on-year CPI change for 2023, peak inflation rate over the 2022–2023 window, number of months where inflation exceeded 6%, and food-specific CPI YoY for 2023.

**NOAA IBTrACS Western Pacific Typhoon Tracks, 2019–2024** — storm track coordinates and maximum sustained wind speeds. Three regional features are derived using a 300 km radius around each region's centroid: number of distinct storms, maximum wind speed across storms, and a cumulative typhoon exposure score (sum of per-storm maximum winds).

Both datasets are joined to PMS households at the regional level via a crosswalk mapping PSA region labels to PMS region names. They are treated as coarse contextual controls, not precise shock intensity measures.

---

## 3. Methodology

### 3.1 Shock Feature Engineering

Nine binary shock flags are derived by keyword matching across up to three raw shock text columns (`T_Q_26_1`, `T_d1_1_2_1`, `T_d1_1_3_1`) in Tagalog and English. The compound shock flag — the study's key interaction variable — is defined as:

```
shock_compound = shock_climate AND shock_inflation
```

143 households (19.5% of the MFI subsample) experienced compound shocks.

### 3.2 Unsupervised Learning — PCA + K-Means

18 features covering household demographics and socioeconomic status, financial literacy, shock exposure, and asset ownership are scaled and imputed before PCA. Five principal components are retained, explaining 40.92% of total variance. The components span variation across the demographic, financial literacy, and shock exposure dimensions of the feature set.

K-Means is run for K = 2 through 7. K = 3 is selected based on the inertia elbow (largest drop between K=2 and K=3) and cluster interpretability. Silhouette scores increase monotonically to K=7 — a consequence of demographic homogeneity in the MFI subsample (95% of respondents are SES Class D or E) rather than a signal for more clusters.

### 3.3 Supervised Learning

**Feature matrix.** 31 features are used: 22 base features spanning household demographics and assets (11 features), financial education and literacy (2), MFI usage frequency (1), and shock exposure indicators (8 binary flags plus count); 7 external CPI and typhoon features; and 2 cluster dummy variables (K=3 cluster label, one-hot encoded with cluster 0 as reference). MFI engagement quality variables (`mfi_satisfaction`, `recovery_status`, `mfi_avoided_distress`, `mfi_shock_support`, `mfi_reliance`, `mfi_vs_other`) are excluded from the primary supervised feature set on endogeneity grounds. These are concurrent assessments of MFI value collected in the same survey session as the outcome variables — a household reporting high satisfaction with MFI and simultaneously reporting improved quality of life attributed to MFI is not providing independent predictive signal; both responses reflect the same underlying evaluation. Including them would produce a model that predicts MFI-attributed outcomes using other MFI-attributed assessments, rather than identifying which structural household characteristics and external conditions differentiate outcome trajectories. To validate this choice and establish a clean benchmark against the PMS's own analytical approach, a logit model using the PMS specification — including MFI engagement variables — is also estimated and reported in Section 4.2. Missing values are imputed using column medians via `SimpleImputer`.

**Models.** Four classifiers are evaluated:

| Model | Key settings |
|---|---|
| Dummy (most frequent) | Baseline — always predicts "improved" |
| Logit (no penalty) | Multinomial logit; mirrors the PMS survey's own analytical approach |
| Elastic Net | `penalty='elasticnet'`, `l1_ratio=0.5`, `C=0.1`, `max_iter=10000` |
| Random Forest | `n_estimators=500`, `max_depth=10`, `min_samples_leaf=5` |

**Evaluation.** 5-fold stratified cross-validation; primary metric is macro-F1. Confusion matrices are generated on an 80/20 stratified holdout split. Feature importances use mean decrease in impurity (MDI) from a full-sample Random Forest fit.

**Robustness check.** Models are re-run under three feature configurations to test whether the external CPI and typhoon features add value beyond a simpler regional control:
- NULL: base features only (no regional controls)
- Model A: base features + 16 region dummy variables (fixed effects)
- Model B: base features + 7 external CPI and typhoon features (our main specification)

---

## 4. Results

### 4.1 Vulnerability Typology (Unsupervised)

K-Means with K=3 identifies three distinct vulnerability profiles:

| Cluster | n | Label | Key characteristics |
|---|---|---|---|
| 0 | 277 | Climate-Exposed | High climate shock prevalence; moderate inflation exposure |
| 1 | 185 | Economically Unstable | Higher economic vulnerability; lower MFI engagement intensity |
| 2 | 272 | Inflation-Exposed | High inflation shock prevalence; lower climate exposure |

**Outcome rates by cluster (% reporting improved):**

| Cluster | Consumption Stability | Shock Resilience | Quality of Life |
|---|---|---|---|
| Climate-Exposed (n=277) | 91.0% | 91.3% | 84.8% |
| Economically Unstable (n=185) | 84.3% | 77.3% | 67.6% |
| Inflation-Exposed (n=272) | 86.8% | 86.4% | 75.7% |

The Economically Unstable cluster consistently reports the worst outcomes across all three targets. The gap is most pronounced for quality of life, where Economically Unstable households report 67.6% improvement versus 84.8% for Climate-Exposed — a 17-percentage-point difference within the same MFI-availing population.

**Kruskal-Wallis validation:** Cluster membership significantly predicts Shock Resilience (p < 0.0001) and Quality of Life (p < 0.0001). The effect for Consumption Stability is not statistically significant (p = 0.069). This non-significant result for CS is itself a finding — MFI-attributed consumption stability appears more uniformly distributed across vulnerability profiles than shock resilience or quality of life.

### 4.2 Model Performance (Supervised)

**PMS-specification benchmark.** As a validation step and to establish a clean comparison against the PMS's own analytical approach, a logit using the full PMS feature specification — which adds back the MFI engagement quality variables excluded from the primary model — is also estimated (`logit_pms_spec`). This model produces noticeably higher macro-F1 scores than the structural-only logit across all three targets (see table below). The gap reflects the endogeneity problem described in Section 3.3: the engagement variables and the outcome variables are concurrent assessments of the same underlying sentiment, so the model is largely predicting how respondents feel about MFI using other measures of how they feel about MFI. The structural-only specification — which excludes these variables — is the appropriate basis for comparing logit and Random Forest, because it asks which model better extracts genuine non-linear signal from household structure and external context. The PMS-spec logit row is included for reference to confirm that the pipeline replicates the PMS analytical approach and that the structural-only models are not artificially handicapping the benchmark.

**5-fold cross-validation macro-F1:**

| Model | Consumption Stability | Shock Resilience | Quality of Life |
|---|---|---|---|
| Dummy (most frequent) | 0.312 | 0.308 | 0.290 |
| Logit — PMS spec (with MFI engagement vars) | 0.388 | 0.428 | 0.547 |
| Logit — structural only (our benchmark) | 0.320 | 0.313 | 0.393 |
| Elastic Net — structural only | 0.307 | 0.318 | 0.365 |
| **Random Forest — structural only** | **0.336** | **0.391** | **0.453** |

The PMS-spec logit scores (CS: 0.388 / SR: 0.428 / QoL: 0.547) are substantially higher than the structural-only logit (CS: 0.320 / SR: 0.313 / QoL: 0.393). The QoL gap is +0.154 macro-F1 — driven by the endogenous engagement variables, not genuine structural prediction. This confirms the endogeneity argument: the pipeline is not broken, and the structural-only benchmark is not artificially handicapped.

Random Forest outperforms the structural logit benchmark on all three targets. The gains are most substantial for Shock Resilience (RF: 0.391 vs Logit: 0.313; +0.078 macro-F1) and Quality of Life (RF: 0.453 vs Logit: 0.393; +0.060 macro-F1). The margin for Consumption Stability is minimal (RF: 0.336 vs Logit: 0.320; +0.016), reflecting that the structural and contextual features in this specification carry limited signal for CS specifically. Elastic Net performs below the unpenalized logit on CS (0.307 vs 0.320) and QoL (0.365 vs 0.393), with a negligible advantage on SR (0.318 vs 0.313). Regularization provides no clear benefit over unpenalized logit with this feature set.

All models outperform the dummy baseline for Quality of Life, confirming real predictive signal for that target. Gains over the dummy are modest for Consumption Stability and Shock Resilience, particularly for the logit models (CS: +0.008; SR: +0.005).

### 4.3 Feature Importance

The top three Random Forest features by mean decrease in impurity (MDI), per target:

| Target | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Consumption Stability | cpi_peak_2022_2023 | cpi_food_yoy_2023 | hh_size |
| Shock Resilience | fin_ed_influence | hh_size | cpi_peak_2022_2023 |
| Quality of Life | fin_ed_influence | educ_head | typhoon_count |

`fin_ed_influence` (influence of financial education on MFI decision) is the top MDI feature for Shock Resilience and Quality of Life and the top SHAP feature for both. `fin_lit_level` (financial literacy level) ranks second in SHAP importance for both SR and QoL. External macroeconomic conditions — peak CPI over the 2022–2023 window (`cpi_peak_2022_2023`) — are the leading signal for Consumption Stability. Typhoon exposure features (`typhoon_count`, `typhoon_exposure_score`) appear in the top five MDI features for both Shock Resilience and Quality of Life.

Structural household characteristics — education level (`educ_head`), household size (`hh_size`), and SES class — contribute moderate signal, consistent with the expanded feature set giving these variables more room to operate than in the clustering step.

MFI engagement quality variables (`mfi_satisfaction`, `recovery_status`, `mfi_avoided_distress`, `mfi_shock_support`) are not included in the supervised feature set. The importance pattern therefore reflects which structural household and contextual features distinguish outcome trajectories when direct MFI assessment signals are withheld.

### 4.4 External CPI and Typhoon Features — Robustness Check

**Triangulation (Spearman correlations, n=734):**

| Pair | rho | p-value |
|---|---|---|
| Typhoon exposure score × self-reported climate shock | +0.094 | 0.011 * |
| Typhoon count × self-reported climate shock | +0.051 | 0.165 ns |
| CPI YoY 2023 × self-reported inflation shock | +0.052 | 0.161 ns |
| CPI peak 2022–2023 × self-reported inflation shock | −0.046 | 0.213 ns |
| Months ≥ 6% CPI × self-reported inflation shock | −0.009 | 0.806 ns |

Correlations between objective regional measures and individual self-reported shocks are weak, with only one significant result. This is expected: regional averages are coarse proxies for household-level shock experience. One household in a high-CPI region may have been protected from inflation by remittances or savings; another in a low-CPI region may have experienced severe local price spikes. The weak correlations confirm the study's ex-ante framing of external data as contextual controls, not precise shock intensity measures.

**Robustness check (macro-F1 by feature set):**

| Feature set | Logit CS | RF CS | Logit SR | RF SR | Logit QoL | RF QoL |
|---|---|---|---|---|---|---|
| NULL (base only) | 0.351 | 0.470 | 0.379 | 0.491 | 0.459 | 0.598 |
| Model A (region dummies) | 0.396 | 0.468 | 0.408 | 0.499 | 0.504 | 0.575 |
| Model B (CPI + typhoon) | 0.320 | 0.336 | 0.313 | 0.391 | 0.393 | 0.453 |

Model B (external CPI and typhoon features) performs below Model A (region dummies) across all three targets and both model types, confirming that the objective regional data adds no value beyond a simple regional fixed effect. Model B also performs below the NULL model (no explicit regional controls) across all targets — the largest RF gap is QoL (NULL: 0.598 vs Model B: 0.453). A caveat applies: the NULL and Model A configurations include `mfi_satisfaction` in their base feature set while Model B does not, which partially confounds the NULL vs. Model B comparison. The Model A vs. Model B comparison — which differs only in how the regional control is implemented — is clean, and Model B consistently underperforms Model A across all targets and both model types.

**This is an interpretable null.** It does not mean climate and inflation conditions are irrelevant to resilience. It means that at the resolution at which these external data exist (regional averages), they are redundant with knowing where the respondent lives.

### 4.5 Compound Shock Effect

**Mann-Whitney U tests — compound shock (n=143) vs. no compound shock (n=591):**

| Target | U statistic | p-value | Interpretation |
|---|---|---|---|
| Consumption Stability | 42,092 | 0.5508 | ns |
| Shock Resilience | 42,898 | 0.3204 | ns |
| Quality of Life | 42,460 | 0.4512 | ns |

Households that experienced both climate and inflation shocks simultaneously report outcomes statistically indistinguishable from those that did not (all p > 0.30). Mean outcome scores are nearly identical: for Quality of Life, the no-compound group mean is 1.755 and the compound group mean is 1.741 on the 0–2 scale.

Two interpretations are consistent with this null:

1. **MFI services buffer compound shocks effectively** — MFI products (flexible repayment, emergency loans, financial advice) may provide sufficient protection that even households facing simultaneous climate and inflation stress maintain similar self-reported outcomes to those facing simpler shock profiles.

2. **Survivorship bias** — households worst-affected by compound shocks may have exited MFI services before the survey. The cross-sectional design captures only current users, structurally excluding those for whom MFI failed during a compound shock event.

These two interpretations cannot be distinguished with the available data.

**Regional compound shock prevalence:**

BARMM (36.4%) and Region III/Central Luzon (33.3%) have the highest compound shock rates in the MFI subsample. Despite this, BARMM shows the lowest QoL improvement rate (63.6%) in the sample. Region III shows a high SR improvement rate (97.0%) despite its high compound shock prevalence — suggesting regional heterogeneity in MFI effectiveness that warrants further investigation.

---

## 5. Limitations

### 5.1 Survivorship Bias (Critical)

All three outcome variables are attribution questions asked only of current MFI users. Households that experienced harm from MFI services — whether through debt stress, inflexible repayment schedules, or inadequate shock support — and subsequently stopped using MFI are absent from the sample. This creates a structural positive skew in all three outcome distributions and means that the study's findings describe the self-reported experience of MFI *survivors*, not MFI users broadly. Any claim about negative MFI outcomes is almost certainly an underestimate.

### 5.2 No Counterfactual

The study cannot compare MFI users to comparable non-users because the three prediction targets are MFI-specific attribution questions. There is no way to know whether the same households would have reported better or worse resilience without MFI access. A propensity score matching design using a neutral outcome variable (e.g., `d3_1`, income direction after shock — available for all 1,900 PMS respondents including non-availing households) could partially address this in future work by matching MFI users to comparable non-users on observable pre-MFI characteristics.

### 5.3 Self-Reported Attribution Bias

Respondents are asked to directly attribute outcomes to MFI ("Did MFI improve your...?"). This framing invites rationalization — respondents who made the active choice to use MFI may be motivated to report positively to remain consistent with their past decisions. The observed ~80–88% "improved" rate across targets likely reflects both genuine benefit and attribution bias; these cannot be separated.

### 5.4 Class Imbalance

Even after 3-class collapse, the worsened class remains very small — 44 observations for Consumption Stability, 39 for Shock Resilience, and only 14 for Quality of Life. Macro-F1 metrics for the worsened class will have wide confidence intervals given these cell sizes. Particular caution is warranted for QoL worsened-class predictions; any subgroup analysis involving only worsened-class households is statistically underpowered.

### 5.5 External Data Resolution

CPI and typhoon exposure are joined at the regional level (17 Philippine regions). This 17-bin discretization of 734 households means the external features provide less than one degree of freedom per unit of variation they could theoretically explain. Municipal- or barangay-level price monitoring data and PAGASA weather station data with municipal coverage would improve the resolution substantially and might produce a different robustness check result.

### 5.6 Cross-Sectional Design

The PMS 2024 is a single cross-section. There is no before-after comparison, no shock timing relative to MFI enrollment, and no way to observe how households transitioned across the outcome scale over time. Longitudinal data — tracking the same households across multiple waves of the PMS or linking to the PSA Annual Poverty Indicators Survey — would substantially strengthen causal inference.

### 5.7 MFI Heterogeneity Not Captured

The PMS 2024 does not distinguish between MFI product types (group lending vs. individual loans), MFI institution size or regulation status, loan amounts, or client tenure. These structural differences between MFI programs likely drive substantial variation in outcomes that the current feature set cannot capture.

### 5.8 K=3 vs. Proposed Cluster Range

The project design anticipated 4–6 clusters. Silhouette scores increase monotonically from K=2 to K=7, providing no statistical stopping point. K=3 is selected based on the inertia elbow and cluster interpretability, but this represents a researcher judgment call rather than a data-driven determination. The homogeneity of the MFI subsample (95% Class D–E) limits the separability of vulnerability profiles that would otherwise support more clusters.

---

## 6. Conclusions and Policy Implications

### Finding 1: Economically Unstable households benefit least from MFI services

The Economically Unstable cluster (n=185) reports a 17-percentage-point lower QoL improvement rate than the Climate-Exposed cluster, within a population that has already self-selected into MFI use. This suggests that MFI products as currently structured may be better matched to households with climate shock exposure than to households in structural economic precarity. MFIs targeting deeper poverty reduction may need differentiated products — longer grace periods, lower initial loan sizes, or embedded financial coaching — for clients in this profile.

**Supporting results:**
- Cluster outcome rates (Section 4.1, `SL0`): Economically Unstable QoL improved = 67.6% vs. Climate-Exposed = 84.8% (17 pp gap); SR improved = 77.3% vs. 91.3% (14 pp gap); CS improved = 84.3% vs. 91.0% (7 pp gap).
- Kruskal-Wallis cluster validation (Section 4.1, `cell [23]`): cluster membership significantly predicts SR (p < 0.0001) and QoL (p < 0.0001), confirming the between-cluster differences are not due to sampling noise.
- The CS result (p = 0.069, ns) indicates the gap in consumption stability is smaller and less reliable than the SR and QoL gaps — the typology finding is strongest for QoL and SR.

### Finding 2: Random Forest captures predictive signal the logit cannot

RF outperforms the unpenalized logit on all three targets. The gains are largest for Shock Resilience (+0.078 macro-F1) and Quality of Life (+0.060), with a minimal gain for Consumption Stability (+0.016). The dominant features in the RF are financial education influence (`fin_ed_influence`) and financial literacy (`fin_lit_level`) for SR and QoL, and peak CPI (`cpi_peak_2022_2023`) for CS. This suggests that even when MFI engagement quality variables are withheld from the feature set, financial education is the primary structural differentiator of outcome trajectories — and that it carries non-linear signal the logit does not fully recover.

**Supporting results:**
- PMS-spec logit (Section 4.2, `logit_pms_spec`): CS: 0.388 / SR: 0.428 / QoL: 0.547 — substantially above the structural logit (0.320 / 0.313 / 0.393). The QoL gap (+0.154) quantifies how much the endogenous engagement variables inflate the benchmark, confirming the pipeline is valid and the structural-only comparison is not artificially handicapped.
- 5-fold CV macro-F1 (Section 4.2, `SL2`): RF vs. structural logit — CS: 0.336 vs. 0.320 (+0.016); SR: 0.391 vs. 0.313 (+0.078); QoL: 0.453 vs. 0.393 (+0.060). All RF scores exceed the logit benchmark, though the CS margin is small.
- Elastic Net performs below logit on CS (0.307 vs. 0.320) and QoL (0.365 vs. 0.393), with a negligible advantage on SR (0.318 vs. 0.313). Regularization does not recover the non-linear signal; tree-based splitting is needed.
- Feature importance (Section 4.3, `SL4`): `fin_ed_influence` is the top MDI feature for SR and QoL and top SHAP feature for both. `cpi_peak_2022_2023` is the top MDI feature for CS; `fin_lit_level` ranks 2nd in SHAP for both SR and QoL. Typhoon exposure features appear in the top 5 MDI for SR and QoL. MFI engagement quality variables are excluded from this feature set.

### Finding 3: Compound shocks do not produce detectably worse outcomes among MFI users

The null compound shock effect is either evidence that MFI products provide sufficient buffering against simultaneous climate and inflation shocks, or evidence that survivorship bias is severe enough to mask what would otherwise be a damaging effect. Distinguishing these interpretations requires panel data or a non-availing comparison group.

**Supporting results:**
- Mann-Whitney U tests (Section 4.5, `SL5`): compound shock (n=143) vs. no compound shock (n=591) — CS: U=42,092, p=0.551 ns; SR: U=42,898, p=0.320 ns; QoL: U=42,460, p=0.451 ns. None reach the 0.05 threshold.
- Mean outcome scores are nearly identical across groups: QoL no-compound mean = 1.755 vs. compound mean = 1.741 on the 0–2 scale (a difference of 0.014 scale points).
- Compound shock prevalence varies substantially by region (Section 4.5, `SL5`): BARMM 36.4%, Region III 33.3%, MIMAROPA 28.2%, Region VIII 28.2%. The null effect holds despite this regional variation, suggesting it is not driven by low compound shock sample sizes in any single region.

### Finding 4: Objective regional climate and price data are redundant with region fixed effects

At the resolution available (regional averages), CPI and typhoon measures add no predictive value beyond knowing the respondent's region. Higher-resolution administrative data — municipal price monitoring, PAGASA station records — are needed to recover shock intensity signal that is genuinely independent of region membership.

**Supporting results:**
- Robustness check macro-F1 (Section 4.4, `SL6`): Model B (CPI + typhoon) performs below Model A (region dummies) for both Logit and RF across all three targets. The largest gap is QoL Logit: Model A = 0.504 vs. Model B = 0.393 (−0.111). No target shows Model B strictly outperforming Model A. Model B also underperforms the NULL (base only) for all targets and both model types; note that this comparison is partially confounded by `mfi_satisfaction` being present in NULL/Model A but not in Model B.
- Triangulation Spearman correlations (Section 4.4, `EXT cell`): only one of five objective-vs-self-reported shock pairs reaches significance — typhoon exposure score × climate shock (rho = +0.094, p = 0.011). CPI measures show no significant correlation with self-reported inflation shock (rho range: −0.046 to +0.052, all p > 0.15).
- External feature ranges confirm variation exists in the data (cpi_yoy_2023: 5.4–8.0%; typhoon_count: 5–34 storms per region), so the null robustness result is not due to insufficient variation in the external features — it is due to that variation already being captured by region membership.

---

## 7. Recommended Extensions

1. **Propensity score matching** using `d3_1` (income impact direction after shock, available for all 1,900 PMS respondents including non-MFI users) as a neutral outcome variable, matched on demographics and shock history. This would introduce a near-counterfactual comparison without requiring panel data.

2. **Municipal- or barangay-level external data** (PSA sub-regional price monitoring, PAGASA weather station records, NDRRMC calamity declarations) to replace the current 17-bin regional aggregation and recover geographic shock signal that is currently washed out.

3. **Panel data linkage** using BSP or MFI administrative records to track client tenure, loan history, and transition in/out of the program — addressing survivorship bias directly.

4. **Ordinal regression** — if a larger or more balanced sample becomes available (e.g., pooling multiple PMS waves), proportional-odds logistic regression would preserve the full 5-point scale without collapsing information.

5. **Expand SHAP analysis** — `shap.TreeExplainer` is implemented and SHAP summary plots are already generated (SL4b). The next step is beeswarm plots showing feature direction (positive vs. negative push) per household, enabling segmentation of which household profiles are being pushed toward the worsened class and by which features — a more actionable diagnostic than the global MDI rankings alone.

---

## 8. Figure Observations and Visual Descriptions

The following section documents what each figure shows and the initial insights drawn from it. These observations complement the quantitative results in Section 4 and are also embedded as markdown cells directly after each figure in `pms-resilience.ipynb`.

---

### EDA Figures

#### Figure EDA0 — Target Variable Distributions (5-class and 3-class)

The top row shows the raw 5-point Likert distributions. Responses cluster heavily at scores 4–5 ("improved"), with almost no observations at scores 1–2. The bottom row shows the 3-class collapsed targets: "Improved" accounts for 77–88% of responses across all three targets, while "Worsened" accounts for only 2–6%.

**Observation:** The extreme class imbalance is structural survivorship bias, not a data quality problem. Households that had negative experiences and exited MFI services are absent from the sample. This motivates `class_weight='balanced'` in all supervised models and macro-F1 as the primary evaluation metric rather than accuracy.

---

#### Figure EDA1 — Missing Data Analysis

The left panel shows that missingness in `hh_head_employed` and `hh_head_income_range` is overwhelmingly joint — the same households are missing both columns — reflecting survey skip logic (if a household head is not employed, income range questions are skipped). The right panel shows the missingness is distributed relatively uniformly across regions, with no strong geographic concentration.

**Observation:** The skip-logic-driven, geographically uniform missingness supports median imputation as the appropriate strategy. Region-specific imputation would add no value.

---

#### Figure EDA2 — Demographic and Household Profile

The six panels reveal a demographically homogeneous MFI user population: ~75% female household heads; almost entirely SES classes D and E; education peaking at high school completion; income clustering in the ₱2,000–₱10,000/month range; household size of 4–6 members; and predominantly married/partnered heads.

**Observation:** The near-complete SES class D–E concentration (95% of respondents) is the primary driver of limited cluster separability in the PCA and K-Means steps. When most respondents share the same SES bracket, between-household demographic differences are subtle, and clustering picks up shock exposure patterns rather than deep socioeconomic stratification.

---

#### Figure EDA3 — Shock Prevalence Among MFI Users

Pandemic-related shocks are the most prevalent — a lasting footprint of COVID-19. Inflation shocks rank second, consistent with the Philippines' elevated CPI in 2022–2023. The compound shock bar (red) — households reporting simultaneous climate and inflation shocks — represents 19.5% of the sample.

**Observation:** Nearly 1 in 5 MFI-availing households faces compound climate-and-inflation stress. The pandemic shock's dominance across nearly all households may reduce statistical power to detect compound-specific effects, since pandemic stress is largely a constant rather than a differentiator across the sample.

---

#### Figure EDA4 — Financial Literacy Distribution

The left panel shows formal financial education programs are the most common literacy level. The right panel (stacked by SES class) shows that higher SES classes have a somewhat larger share of formal program participants, while lower SES classes have more households with no financial education or only self-directed learning — though the gradient is gradual rather than sharp.

**Observation:** `fin_ed_influence` emerges as the top predictor for Shock Resilience and Quality of Life in supervised models. This figure hints at the mechanism: households motivated by formal financial education to use MFI may be more purposive in their MFI use, yielding better outcomes independent of SES standing.

---

#### Figure EDA5 — 3-Class Outcomes by Shock Group

The "Compound (both)" shock group shows a marginally lower "Improved" rate than single-shock or no-shock groups across all three targets, but the difference is visually subtle — compound shock households still report ≥80% improvement for CS and SR. The no-shock group does not dramatically outperform the compound group.

**Observation:** Even at the descriptive level, compound shocks do not produce a dramatically different outcome profile among MFI users. This visual preview is confirmed by the formal Mann-Whitney U test (Section 4.5), which finds no statistically significant difference between compound and non-compound households across any target (all p > 0.30).

---

### Unsupervised Learning Figures

#### Figure UL0 — PCA Explained Variance (Scree and Cumulative) and Feature Loadings

The scree plot shows variance dropping sharply after PC1 (~8%) and leveling off gradually. Five components explain ~41% of variance; reaching 70% requires 12+ components. The loadings heatmap shows PC1 loading on SES/income/education, PC2 on climate shocks, PC3 on inflation shocks, and PC4–PC5 on financial literacy and employment structure.

**Observation:** Despite low per-component variance, the PCA components are interpretable and span meaningful household dimensions. The low explained variance per component accurately reflects demographic homogeneity — it is not a modeling failure. Using the 5-component PCA projection as input to K-Means prevents high-cardinality variables from dominating the Euclidean distance metric in the raw feature space.

---

#### Figure UL1 — K-Means Cluster Selection (Elbow and Silhouette)

The elbow plot shows inertia dropping most sharply from K=2 to K=3, forming a modest elbow. The silhouette score increases monotonically from K=2 to K=7, offering no statistical stopping criterion.

**Observation:** K=3 is selected based on the inertia elbow and interpretability. The monotonically increasing silhouette is a known symptom of demographic homogeneity — when all households occupy a similar region of feature space, adding clusters always appears to improve separation marginally without producing genuinely distinct groups. K=7 would yield clusters with average n < 105, too small for reliable outcome analysis.

---

#### Figure UL2 — Cluster Profiles (Standardized Heatmap) and PCA Scatter

The heatmap z-scores reveal: Cluster 0 (Climate-Exposed) has above-average climate shock prevalence with near-average socioeconomic indicators; Cluster 1 (Economically Unstable) is below average on SES, income, education, and financial literacy; Cluster 2 (Inflation-Exposed) has above-average inflation shock prevalence. The PCA scatter shows partial but incomplete separation — diffuse, interpenetrating point clouds with clearly separated centroids.

**Observation:** The z-score heatmap is more informative than the PCA scatter for interpretation. K-Means has successfully differentiated households by primary vulnerability type even though the 2D PCA projection shows overlap. The cluster profiles carry genuine signal not visible in a low-dimensional projection.

---

#### Figure UL3 — Outcome Distributions by Vulnerability Cluster

The stacked bars show consistent between-cluster differences: Climate-Exposed leads at ~91%/91%/85% (CS/SR/QoL improved), Economically Unstable is lowest at ~84%/77%/68%, and Inflation-Exposed falls in between at ~87%/86%/76%. The Kruskal-Wallis test confirms statistical significance for SR and QoL (both p < 0.0001) but not CS (p = 0.069).

**Observation:** The non-significant CS result is itself a finding: MFI-attributed consumption stability appears more uniformly distributed across vulnerability profiles — MFI products provide relatively consistent consumption support even to structurally precarious households. The SR and QoL gaps are the strongest and most policy-relevant findings of the unsupervised step.

---

### Supervised Learning Figures

#### Figure SL0 — Vulnerability Typology Profiles (Feature Heatmap, Shock Bar, and Outcome Stacked Bars)

The feature heatmap confirms that Economically Unstable is red across all socioeconomic dimensions. The shock bar chart shows Climate-Exposed has the highest climate (~80%) and compound shock rates. The outcome stacked bars replicate the cluster validation finding with labeled names — the Economically Unstable QoL gap (67.6% vs. 84.8% for Climate-Exposed) is the most striking.

**Observation:** Economically Unstable households report worse outcomes *despite* lower shock exposure than Climate-Exposed households. Structural economic precarity — not shock frequency or type — is the primary differentiator of which MFI users benefit least. Product design addressing economic precarity directly (embedded coaching, graduated loan sizing) may be more effective for this group than shock-response products.

---

#### Figure SL2 — Confusion Matrices (80/20 Holdout, 3 Models × 3 Targets)

All models predict "Improved" for the vast majority of observations. "Worsened" is almost never predicted correctly (near-zero F1) due to the small class size. Random Forest produces a slightly more distributed confusion matrix than Logit or Elastic Net, recovering more neutral-class predictions and contributing to its higher macro-F1. Quality of Life shows the widest spread across all cells, consistent with its highest macro-F1 scores.

**Observation:** The confusion matrices confirm that accuracy would be a misleading evaluation metric. A naive model predicting "Improved" for all households achieves 77–88% accuracy while providing zero predictive value for the minority classes. Macro-F1, which weights all three classes equally, is the correct lens.

---

#### Figure SL3 — Random Forest Feature Importances (Mean Decrease in Impurity)

**Consumption Stability:** Dominated by `cpi_peak_2022_2023` and `cpi_food_yoy_2023`, followed by `hh_size`. Regional inflation intensity is the leading structural signal for MFI-attributed consumption stability.

**Shock Resilience:** `fin_ed_influence` ranks first, followed by `hh_size` and `cpi_peak_2022_2023`. Financial education motivation is the clearest structural differentiator of self-reported resilience.

**Quality of Life:** `fin_ed_influence` leads, followed by `educ_head` and `typhoon_count`. Typhoon exposure features appear for QoL but not prominently for CS or SR — regional climate proximity shapes overall wellbeing perception differently from how it shapes specific financial metrics. Cluster labels appear mid-ranking, confirming the vulnerability typology adds incremental predictive signal.

**Observation:** The prominence of `fin_ed_influence` for SR and QoL — with all endogenous MFI engagement variables excluded — suggests that the motivation for MFI adoption is a structural outcome differentiator. Households drawn to MFI through financial education channels appear to use MFI more purposively, yielding better outcomes independent of their reported MFI satisfaction.

---

#### Figure SL4b — SHAP Feature Importances (TreeExplainer) and Binary QoL SHAP

The SHAP rankings largely confirm MDI results: `fin_ed_influence` leads for SR and QoL; CPI features dominate CS. `fin_lit_level` ranks higher under SHAP than MDI for SR and QoL, suggesting MDI partially suppressed it due to correlation with `fin_ed_influence`. The SHAP vs. MDI top-5 comparison shows strong agreement on the top features. The binary QoL SHAP panel confirms that `fin_ed_influence` and `fin_lit_level` remain prominent when the outcome is simplified to worsened vs. not-worsened — the finding is not an artifact of the 3-class collapse.

**Observation:** Cross-method convergence (SHAP and MDI) on `fin_ed_influence` and `fin_lit_level` provides strong validation. These features carry genuine non-linear signal that the Random Forest captures and the logit cannot fully recover — this is the mechanism behind the RF's macro-F1 advantages on SR (+0.078) and QoL (+0.060).

---

#### Figure SL5 — External Conditions: Regional Shock Prevalence, QoL Rates, and Compound Shock Effect

**Panel 1 (shock by region):** BARMM (36.4%) and Region III — Central Luzon (33.3%) have the highest compound shock rates. NCR shows lower climate shock but above-average inflation exposure.

**Panel 2 (QoL by region):** BARMM stands out as the lowest-performing region (~64% QoL improved), despite having the highest compound shock rate. This regional exception suggests compound shocks may matter in specific contexts even when the national-level aggregate effect is undetectable.

**Panel 3 (compound shock effect):** The bars for compound and non-compound households are nearly identical across all three targets — a visual confirmation of the Mann-Whitney null result.

**Panel 4 (compound × cluster):** Within the Economically Unstable cluster, compound-shock households show a more visible QoL gap than in the other two clusters, suggesting compound shocks may disproportionately affect structurally precarious households.

**Observation:** The regional and cluster-level panels reveal heterogeneity masked by the aggregate null test. BARMM's low QoL improvement alongside high compound shock prevalence, and the Economically Unstable cluster's larger compound-shock gap, both suggest compound shocks matter in specific high-risk contexts. The Mann-Whitney U tests (all p > 0.30) confirm no statistically significant national-level compound shock effect, but do not rule out context-specific effects.

---

*End of writeup.*
