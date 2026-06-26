# Pre-EDA Technical Documentation
## MFI Study — AI 221 Classical Machine Learning Project
**Author:** Quiwa, Quiel Andrew I.
**Last updated:** 2026-05-27
**Notebook:** `pms-resilience.ipynb`

---

## 1. Project Overview

This project applies classical ML methods to the **Philippine Microfinance Survey 2024 (PMS 2024)** to analyze whether MFI use is associated with better household resilience outcomes following economic and climate shocks. The study is restricted to the 734 MFI service users in the sample.

**Three prediction targets** (from PMS Figures 7–9):
| Target column | What it measures |
|---|---|
| `consumption_stability` | Did MFI improve ability to maintain stable household spending? |
| `shock_resilience` | Did MFI strengthen resilience to economic downturns? |
| `quality_of_life` | Did MFI improve overall household wellbeing? |

All three are **self-reported attribution questions** — respondents evaluate the effect of MFI on their household, not an objective measure of income or assets. This framing is important for interpreting results (see Section 8).

**Two modeling steps planned:**
1. Unsupervised: PCA + K-Means clustering for vulnerability profiling. PCA uses 18 features covering household demographics and socioeconomic status, financial literacy, shock exposure, and asset ownership. MFI engagement quality variables are not included in the PCA feature set.
2. Supervised: Elastic Net and Random Forest classifiers, benchmarked against the PMS multinomial logit baseline. The supervised feature set (31 features) uses the same demographic, financial literacy, and shock features plus external regional data and cluster dummies. MFI engagement quality variables (`mfi_satisfaction`, `recovery_status`, `mfi_avoided_distress`, `mfi_shock_support`, `mfi_reliance`, `mfi_vs_other`) are excluded from the supervised feature set.

**External datasets to be joined (at modeling stage):**
- PSA Consumer Price Index for the bottom 30% of households, monthly 2019–2023 (by region)
- NOAA IBTrACS typhoon track data, Western Pacific basin, 2019–2024 (by region)

Both are linked at the regional level and treated as coarse contextual controls, not precise shock intensity measures.

---

## 2. Raw Data and Filtering

**Source file:** `PMS2024_raw.xlsx`
**Raw shape:** 1,900 rows × 1,210 columns

**MFI user filter:** Column `e3_1` (renamed `mfi_availed`) == `'AVAILED'`
**Filtered shape:** 734 rows × 33 columns (before feature engineering)

The filter is applied before any other transformations. This ordering is intentional — see Section 4 (Sentinel Value Handling) for why it matters.

---

## 3. Column Selection and Mapping

33 columns were selected from the 1,210-column raw file. All were renamed from raw survey codes to readable names. Five asset and employment columns were added after a feasibility review against the project proposal (see Section 6 for encoding decisions).

| Raw column | Renamed to | Category | Description |
|---|---|---|---|
| `Q_2` | `region` | Demographics | Region (full text, e.g. "NATIONAL CAPITAL REGION") |
| `a5` | `sex_head` | Demographics | Sex of household head |
| `a9` | `educ_head` | Demographics | Education attainment (11 levels) |
| `Q_107` | `ses_class` | Demographics | SES/MORES class |
| `Q_106` | `civil_status` | Demographics | Civil/marital status |
| `b1_1` | `hh_size` | Demographics | Total household size |
| `I_1_b5` | `hh_head_employed` | Demographics | Household head currently employed |
| `I_1_b6` | `hh_head_income_range` | Demographics | Household head income bracket |
| `Q_109` | `num_working_members` | Demographics | Count of working household members (0–9) |
| `I_1_b8` | `hh_head_employment_type` | Demographics | Employment type of household head |
| `I_1_b14` | `hh_head_remittance` | Demographics | Household head receives remittances (Yes/No) |
| `g1` | `owns_land` | Assets | Household owns land (Yes/No) |
| `Q_116` | `house_quality` | Assets | House wall construction material (6-level ordinal wealth proxy) |
| `e3_1` | `mfi_availed` | MFI usage | MFI availed flag (filter variable, dropped after filtering) |
| `e6` | `mfi_use_frequency` | MFI usage | Frequency of MFI use |
| `e9` | `mfi_reliance` | MFI usage | Degree of reliance on MFI |
| `e11` | `mfi_vs_other` | MFI usage | MFI effectiveness vs. other financial strategies |
| `e12` | `recovery_status` | MFI usage | Recovery status after last shock |
| `e13` | `mfi_avoided_distress` | MFI usage | Whether MFI helped avoid asset sale / informal loans |
| `e17` | `mfi_shock_support` | MFI usage | Type of MFI support received during a shock |
| `e18` | `mfi_business_context` | MFI usage | Had a business before MFI (collapsed to binary `had_business_pre_mfi`) |
| `e20` | `mfi_satisfaction` | MFI usage | Overall MFI satisfaction |
| `f14` | `fin_lit_level` | Financial literacy | Financial literacy level (primary FL variable) |
| `f4` | `fin_ed_influence` | Financial literacy | Influence of financial education on MFI decision |
| `f5` | `has_bank_account` | Financial literacy | Has a formal bank account |
| `T_Q_26_1` | `shock_type_1` | Shocks | Shock type, slot 1 |
| `T_d1_1_2_1` | `shock_type_2` | Shocks | Shock type, slot 2 |
| `T_d1_1_3_1` | `shock_type_3` | Shocks | Shock type, slot 3 |
| `d3_1` | `shock_income_impact_1` | Shocks | Income impact direction, shock 1 |
| `d3_2` | `shock_income_impact_2` | Shocks | Income impact direction, shock 2 |
| `d3_3` | `shock_income_impact_3` | Shocks | Income impact direction, shock 3 |
| `e7` | `consumption_stability` | Targets | 5-point Likert (see Section 7) |
| `e15` | `shock_resilience` | Targets | 5-point Likert (see Section 7) |
| `e16` | `quality_of_life` | Targets | 5-point Likert (see Section 7) |

---

## 4. Sentinel Value Handling

**Decision:** Replace `-1` (string and integer) with `NaN` after filtering to MFI users, not before.

**Reason:** `-1` is a survey skip-logic marker meaning "this question was not shown to this respondent." Among the full 1,900-row sample, many `-1` values in the `e`-series columns legitimately mean "non-MFI user — question not applicable." Replacing `-1` globally before filtering would corrupt these rows by turning a structural skip into a missing value.

By filtering to MFI users first, all remaining `-1` values reflect a different set of skip conditions:
- **e-series sub-questions:** A follow-up filter within the MFI section (e.g., `mfi_reliance` is only shown to respondents who reported using MFI regularly)
- **Shock columns:** `-1` means fewer than 3 shocks were reported; that slot is unused
- **Income range / employed:** `-1` means the respondent was not the household head or the question was skipped

**Result after sentinel replacement (MFI users, n=734):**

| Column | NaN count | % | Source |
|---|---|---|---|
| `hh_head_income_range` | 185 | 25.2% | Skip logic |
| `hh_head_employed` | 167 | 22.8% | Skip logic |
| `shock_income_impact_3` | 90 | 12.3% | Unused shock slot |
| `mfi_reliance` | 88 | 12.0% | Sub-question skip |
| `shock_income_impact_2` | 25 | 3.4% | Unused shock slot |
| `shock_income_impact_1` | 4 | 0.5% | Unused shock slot |

All other columns had 0 NaN at this stage.

---

## 5. Shock Feature Engineering

The three raw shock type columns (`shock_type_1/2/3`) contain full Tagalog/English text for up to three shock events per respondent. Eight binary indicator flags were derived via keyword matching across all three slots simultaneously.

| New column | Keyword(s) matched | Links to external data? |
|---|---|---|
| `shock_typhoon` | `"Bagyo"` | IBTrACS |
| `shock_flood` | `"Pagbaha"` | IBTrACS |
| `shock_climate` | typhoon OR flood | IBTrACS |
| `shock_inflation` | `"pagtataas na presyo"` | PSA CPI |
| `shock_pandemic` | `"Pandemya"` | — |
| `shock_illness` | `"karamdaman"` | — |
| `shock_job_loss` | `"Kawalan ng trabaho"` | — |
| `shock_business` | `"Pagkalugi\|Paghina ng kita\|Hindi naging tagumpay"` | — |

**Compound shock flag:** `shock_compound = shock_climate AND shock_inflation` — the key variable for the compound shock effect finding in the project design (n=143, 19.5% of MFI users).

**`num_shocks`:** Count of non-null, non-"Wala (None)" entries across the three shock slots. Respondents who reported `"Wala (None)"` in a slot explicitly indicated no shock, so those are subtracted from the raw notna() count.

**Prevalence summary (n=734):**
| Shock type | n | % |
|---|---|---|
| Pandemic | 597 | 81.3% |
| Inflation | 401 | 54.6% |
| Climate (any) | 355 | 48.4% |
| Typhoon | 284 | 38.7% |
| Business | 155 | 21.1% |
| Compound (climate + inflation) | 143 | 19.5% |
| Flood | 152 | 20.7% |
| Job loss | 85 | 11.6% |
| Illness | 78 | 10.6% |

---

## 6. Ordinal Encoding Decisions

### Strategy
- Genuinely ordinal columns → integer mapping reflecting natural order
- Binary columns → 0/1
- Nominal columns (`region`, `civil_status`) → left as raw text; will be one-hot encoded at the modeling stage
- "Not applicable", "Others / specify", and "Refused" responses → `NaN` (not ordereable), with specific exceptions noted below

### Column-level decisions

**`educ_head`** — 11-level ordinal (0=none → 10=postgrad). Straightforward ladder ordering. No ambiguous cases.

**`ses_class`** — 7-level ordinal. The MORES classification used in the PMS includes levels E, D, Broad C, C, Upper C, B, A. The map covers all seven. Any value outside this set becomes NaN (1 row found; likely a data entry anomaly).

**`mfi_use_frequency`** — 5-level ordinal (1=less than once a year → 5=more than once per quarter). `"Others (specify)"` (n=106) and `"Refused"` (n=1) are mapped to NaN — they cannot be placed on the frequency scale.

**`fin_lit_level`** — 4-level ordinal (0=none → 3=formal program). `"Iba pa, pakitukoy (Others, specify)"` (n=13) mapped to NaN — unclassifiable literacy source.

**`mfi_reliance`** — 4-level ordinal (0=not at all → 3=heavily). 88 NaN inherited from sentinel replacement (skip logic).

**`mfi_vs_other`** — 3-level ordinal (0=less effective → 2=more effective). `"Not applicable / Have not used other strategies"` (n=6) mapped to NaN — respondents with no comparison basis cannot be placed on the scale.

**`recovery_status`** — 5-level ordinal (0=did not recover → 4=fully recovered). No NaN.

**`mfi_avoided_distress`** — 3-level ordinal (0=did not avoid → 2=avoided asset sale).
> **Decision:** The `"Hindi pa nakaranas ng kahirapang pinansyal (No experience of financial distress)"` response (n=18) was initially mapped to NaN. These 18 households never faced financial distress, so they trivially did not need MFI protection. Treating them as NaN would silently drop them from models. They are recoded to **0** (did not avoid distress via MFI) via `.fillna(0)` after the map, placing them in the same bucket as households that experienced distress but were not helped.

**`mfi_shock_support`** — 4-level ordinal (0=no support offered → 3=emergency loan).
> **Decision:** Confirmed single-select via `value_counts()` — no comma-joined combination responses exist in the data. The ordinal ordering (flexible repayment < financial advice < emergency loan) represents increasing intensity of MFI intervention during a shock. `"Did not seek support"` (n=66) is kept as NaN: this is a distinct behavioral state (the household chose not to engage MFI during a shock) that is not equivalent to "no support was offered."

**`mfi_satisfaction`** — 5-level ordinal (0=very dissatisfied → 4=very satisfied). Note: this uses a 0–4 scale vs. the targets' 1–5 scale. Both are internally consistent; do not conflate in visualizations.

**`had_business_pre_mfi`** — binary derived from `mfi_business_context` (e18). `"Wala"` in the response string → 0 (no prior business); any other non-null value → 1. The original text column is dropped.

**`hh_head_income_range`** — 10-level ordinal bracket rank (0=below 1,000 → 9=40,001 to 50,000). `"Refused"` (n=2) mapped to NaN.

> **Data quality note:** The initial map used bracket labels assumed from the project design (`"Less than 3,000"`, `"40,001 to 60,000"`, `"More than 60,000"`). These strings do not exist in the actual data. The actual lower-end brackets are `"Below 1,000"`, `"1,001 to 2,000"`, and `"2,001 to 3,000"`, and the upper bracket present in the MFI subsample is `"40,001 to 50,000"`. Using the wrong strings caused 70 additional NaN. Corrected after diagnostic.

**`num_working_members`** — already numeric (integer count 0–9 in raw data). No mapping required; `pd.to_numeric(..., errors='coerce')` applied for type safety. No NaN. This is the actual count of currently working household members, not just the head — a better measure of household labor capacity than the binary `hh_head_employed`.

**`hh_head_employment_type`** — 5-level formality/stability gradient (0=unpaid → 4=government). The ordering reflects income precariousness: unpaid family labor is most vulnerable, self-employment without employees is informal, private wage work adds some stability, and government employment is most secure.

| Level | Employment type(s) |
|---|---|
| 0 | Unpaid worker in family-operated farm or business |
| 1 | Self-employed with no paid employee |
| 2 | Wage worker in private household or family business |
| 3 | Employer in own business; wage worker in private establishment; job-order govt |
| 4 | Permanent government worker; military/police |

167 NaN from sentinel replacement — the same structural skip affecting `hh_head_employed` (households where HH member 1 employment was not recorded).

**`hh_head_remittance`** — binary (0=No, 1=Yes). 103/734 (14%) receive remittances. No NaN.

**`owns_land`** — binary (0=No, 1=Yes). 157/734 (21%) own land. No NaN.

**`house_quality`** — 6-level ordinal (0=salvaged → 5=strong materials). Preferred over `h1` (owns house, 85% Yes) as a housing wealth proxy because it has more distributional spread.

| Level | Wall material |
|---|---|
| 0 | Salvaged/makeshift |
| 1 | Mixed, predominantly salvaged |
| 2 | Light materials (bamboo, sawali, cogon, nipa, anahaw) |
| 3 | Mixed, predominantly light |
| 4 | Mixed, predominantly strong |
| 5 | Strong materials (galvanized iron, concrete, wood, etc.) |

2 NaN from `"Not applicable"` responses.

> **Design note — why these five columns:** The project proposal explicitly names asset ownership, remittances, and number of working members as household profile features. The original COLUMN_MAP omitted them. `owns_land` and `house_quality` provide the asset dimension needed for the "asset-backed vs. asset-poor" clustering axis described in the proposal. `hh_head_employment_type` captures income source structure better than the binary employed flag alone. These features are expected to contribute to PCA/clustering more than to supervised outcome prediction, consistent with EDA findings showing near-zero Spearman correlations for demographic variables.

**`shock_income_impact_1/2/3`** — 3-level ordinal (0=decreased → 2=increased). NaN inherited from sentinel replacement (unused shock slots).

---

## 7. Target Variable Decisions

### 5-class encoding (retained for EDA)
All three targets use the same scale:

| Score | Meaning |
|---|---|
| 1 | Significantly worsened |
| 2 | Somewhat worsened |
| 3 | No change / neutral |
| 4 | Somewhat improved |
| 5 | Significantly improved |

**Distributions (n=734):**

| Score | consumption_stability | shock_resilience | quality_of_life |
|---|---|---|---|
| 1 | 13 | 14 | 6 |
| 2 | 31 | 25 | 8 |
| 3 | 46 | 64 | 154 |
| 4 | 360 | 477 | 450 |
| 5 | 284 | 154 | 116 |

### 3-class collapse (used for modeling)
**Decision:** Collapse to 3 classes — worsened (0), neutral (1), improved (2).

**Mapping:** `{1→0, 2→0, 3→1, 4→2, 5→2}`

**Reason:** The 5-class versions have critically sparse "worsened" categories (6–14% combined). A classifier trained on these distributions will fail to learn the worsened class — it can achieve 77–88% accuracy by predicting "improved" for every row. Collapsing to 3 classes creates a usable worsened bucket while preserving the directional distinction the project design is testing (MFI made things worse / no change / better).

**3-class distributions:**
| Class | cs_3class | sr_3class | qol_3class |
|---|---|---|---|
| 0 — worsened | 44 (6%) | 39 (5%) | **14 (2%)** |
| 1 — neutral | 46 (6%) | 64 (9%) | 154 (21%) |
| 2 — improved | 644 (88%) | 631 (86%) | 566 (77%) |

**Implication for baseline comparison:** The PMS multinomial logit baseline must also be re-run on 3 classes for the comparison to be valid. Keep the 5-class originals for EDA distributions and any descriptive reporting.

---

## 8. Missing Data Registry (After Full Encoding)

All NaN values in the final encoded dataset are documented here. None are silent encoding failures.

| Column | NaN n | NaN % | Type | Treatment |
|---|---|---|---|---|
| `hh_head_income_range` | 187 | 25.5% | 185 skip logic + 2 Refused | Leave as NaN; impute or flag at modeling stage |
| `hh_head_employed` | 167 | 22.8% | Skip logic | Leave as NaN |
| `hh_head_employment_type` | 167 | 22.8% | Skip logic (same source as `hh_head_employed`) | Leave as NaN |
| `mfi_use_frequency` | 107 | 14.6% | 106 "Others/specify" + 1 Refused | Intentional; cannot be ordered |
| `shock_income_impact_3` | 90 | 12.3% | Unused shock slot | Expected; respondent had <3 shocks |
| `mfi_reliance` | 88 | 12.0% | Skip logic | Leave as NaN |
| `mfi_shock_support` | 66 | 9.0% | "Did not seek support" | Intentional; behaviorally distinct from "no support offered" |
| `shock_income_impact_2` | 25 | 3.4% | Unused shock slot | Expected |
| `fin_lit_level` | 13 | 1.8% | "Iba pa / Others, specify" | Intentional; unclassifiable response |
| `mfi_vs_other` | 6 | 0.8% | "Not applicable / no other strategy used" | Intentional; no comparison basis |
| `shock_income_impact_1` | 4 | 0.5% | Unused shock slot (or skip) | Expected |
| `house_quality` | 2 | 0.3% | "Not applicable" response | Intentional; unclassifiable |

Columns with 0 NaN after encoding: all targets, `educ_head`, `ses_class`, `sex_head`, `hh_size`, `num_working_members`, `hh_head_remittance`, `owns_land`, `has_bank_account`, `fin_ed_influence`, `recovery_status`, `mfi_avoided_distress`, `mfi_satisfaction`, `had_business_pre_mfi`, all shock binary flags, `num_shocks`.

---

## 9. Flags and Caveats for EDA and Modeling

### FLAG 1 — qol_3class "worsened" has only 14 rows
`quality_of_life` worsened class (original scores 1+2) has only 14 respondents (2% of sample). Even with `class_weight='balanced'`, any classifier metric reported for this class will have very wide confidence intervals. Report this explicitly in the results. Consider whether binary (improved vs. not improved) is more defensible for `quality_of_life` specifically.

### FLAG 2 — Survivorship bias in all three targets
All three targets are attribution questions asked of current MFI users. Households who experienced harm from MFI and exited the program are not in this sample. This creates a structural positive skew in all three outcome distributions and limits the generalizability of any negative-outcome findings. Acknowledge this as a study limitation.

### FLAG 3 — High missingness in two demographic features
`hh_head_employed` (22.8%) and `hh_head_income_range` (25.5%) have substantial missingness. Before including them in the PCA or classifier feature set, check whether missingness is concentrated in specific groups (e.g., a particular region or SES class). If missingness is non-random, imputation may introduce bias. Plotting a missingness heatmap by region and SES class is recommended as an early EDA step.

### FLAG 4 — mfi_use_frequency has 107 NaN (14.6%) from "Others/specify"
These respondents described a non-standard MFI usage frequency. They are excluded from any analysis using `mfi_use_frequency` as a feature. If this variable becomes important in SHAP, note that 14.6% of users fall outside the standard frequency categories.

### FLAG 5 — mfi_shock_support NaN (66 rows) represents a meaningful behavioral group
The 66 "did not seek support" rows are not missing data — they are households that experienced a shock but chose not to reach out to MFI for help. If `mfi_shock_support` is used as a feature in the classifier, these rows will be excluded from that model's training set. Consider creating a separate binary flag `mfi_sought_support` to preserve this signal without losing rows.

### FLAG 6 — Compound shock group is small but central to the research design
`shock_compound` (climate + inflation simultaneously) has n=143 (19.5%). This is the key group for the compound shock effect finding. Subgroup analysis on this group will have limited statistical power. Report confidence intervals, not just point estimates, for any compound-vs-single-shock comparisons.

### FLAG 7 — New asset/employment features expected to contribute to clustering, not outcome prediction
EDA Spearman correlations showed near-zero associations between demographic variables and all three targets. The five new columns (`num_working_members`, `hh_head_employment_type`, `hh_head_remittance`, `owns_land`, `house_quality`) follow the same demographic pattern and are unlikely to move the needle in supervised feature importance. Their primary value is in PCA/clustering, where they provide an asset and labor capacity axis that was previously absent from the feature matrix. If SHAP analysis confirms near-zero importance in the supervised step, this should be reported as consistent with the EDA finding, not as a modeling failure.

### FLAG 8 — mfi_satisfaction uses 0–4 scale; targets use 1–5 scale
Do not overlay these on the same colorscale or axis in visualizations without relabeling. Both are internally consistent but the different base values will be confusing.

---

## 10. Data Quality Issues Found and Resolved

| Issue | Discovery | Root cause | Resolution |
|---|---|---|---|
| `ses_class` undefined 'E' class | Phantom 'E' key in initial dict, 1 NaN appeared | 'E' and 'Upper C' missing from original map | Added 'Upper C' (rank 5); kept 'E' (rank 1) as valid lowest class; confirmed data contains both |
| `hh_head_income_range` 70 extra NaN | NaN count jumped 185→255 after encoding | Map used assumed bracket labels not present in data ("Less than 3,000", "More than 60,000") | Rebuilt map from actual `value_counts()` output; real labels are "Below 1,000", "1,001 to 2,000", "2,001 to 3,000", "40,001 to 50,000" |
| `mfi_use_frequency` 107 NaN undocumented | NaN appeared with no documented source | 106 "Others (specify)" + 1 "Refused" were not in original map | Added both to map with explicit `np.nan` and comments |
| `fin_lit_level` 13 NaN undocumented | NaN appeared with no documented source | `"Iba pa, pakitukoy (Others, specify)"` string not in map | Added to map with explicit `np.nan` |
| `mfi_avoided_distress` not encoded | Column remained raw text after encoding cell | `.map(avoided_order)` call was missing; only `.fillna(0)` was applied | Fixed to `.map(avoided_order).fillna(0)` — map converts text to integers, then fillna recodes "no experience" NaN to 0 |

---

*End of pre-EDA technical documentation.*
