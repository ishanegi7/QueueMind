# MIMIC-IV-ED Data & Feature Dictionary

This document details raw tables, primary relational keys, and engineered point-in-time features within QueueMind.

---

## 1. Primary Relational Keys

- **`subject_id`** (`int64`): Unique identifier for a unique patient across admissions.
- **`stay_id`** (`int64`): Unique identifier for a single emergency department encounter.
- **`hadm_id`** (`float64`/`int64`, nullable): Inpatient admission identifier (if admitted).

---

## 2. Raw Table Specifications

### 2.1 `edstays` (Master Encounter Table)
| Column | Type | Description | Values / Constraints |
|---|---|---|---|
| `subject_id` | `int64` | Patient identifier | Foreign key |
| `stay_id` | `int64` | ED stay identifier | Primary key (unique) |
| `intime` | `datetime` | Arrival / registration time | Must precede `outtime` |
| `outtime` | `datetime` | Physical departure time | Target-side only; prohibited in features |
| `gender` | `str` | Biological sex | `M`, `F`, `UNKNOWN` |
| `race` | `str` | Self-reported ethnicity | Categorical |
| `arrival_transport` | `str` | Transport mode to ED | `WALK IN`, `AMBULANCE`, `OTHER` |
| `disposition` | `str` | Final disposition code | Target-side only; prohibited in features |

### 2.2 `triage` (Initial Assessment)
| Column | Type | Description | Expected Range / Values |
|---|---|---|---|
| `stay_id` | `int64` | ED stay identifier | Foreign key |
| `temperature` | `float64` | Initial body temp (°F) | 85.0 – 108.0 |
| `heartrate` | `float64` | Initial heart rate (bpm) | 20.0 – 260.0 |
| `resprate` | `float64` | Initial respiratory rate | 4.0 – 60.0 |
| `o2sat` | `float64` | Initial pulse oximetry (%) | 50.0 – 100.0 |
| `sbp` | `float64` | Systolic BP (mmHg) | 40.0 – 260.0 |
| `dbp` | `float64` | Diastolic BP (mmHg) | 20.0 – 160.0 |
| `pain` | `float64` | Initial pain score | Standardized 0.0 – 10.0 |
| `acuity` | `float64` | Triage ESI severity | 1 (Resuscitation) to 5 (Non-urgent) |
| `chiefcomplaint`| `str` | Presenting complaint | Standardized lower-case string |

### 2.3 `vitalsign` (Longitudinal Periodic Vitals)
| Column | Type | Description | Constraints |
|---|---|---|---|
| `stay_id` | `int64` | ED stay identifier | Foreign key |
| `charttime` | `datetime` | Observation timestamp | Must be $\le T_{\text{snapshot}}$ to be included |
| `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`, `pain` | `float64` | Periodic vital readings | Filtered by physiological plausibility bounds |

---

## 3. Engineered Features (Step 2 Implementation)

All features listed below are extracted strictly at snapshot timestamp $T_{\text{snapshot}}$.

### 3.1 Patient-Level Features (`patient_features.py`)

| Feature Name | Source | Type | Snapshot-Safe? | Preprocessing & Meaning | Leakage Considerations |
|---|---|---|:---:|---|---|
| `stay_id` | `edstays` | `int64` | Yes | Encounter identifier | Metadata only, excluded from model training |
| `elapsed_time_minutes` | `edstays` | `float64` | Yes | $(T_{\text{snapshot}} - T_{\text{in}})$ in minutes | Uses only $T_{\text{in}}$ and $T_{\text{snapshot}}$; zero future information |
| `gender` | `edstays` | `str` | Yes | Standardized uppercase string (`M`, `F`, `UNKNOWN`) | Known at registration |
| `arrival_transport` | `edstays` | `str` | Yes | Arrival transport category | Recorded upon presentation |
| `acuity` | `triage` | `float64` | Yes | ESI triage level (1.0 to 5.0) | Recorded at triage $\le T_{\text{in}} \le T_{\text{snapshot}}$ |
| `chiefcomplaint_category` | `triage` | `str` | Yes | Mapped clinical category (e.g. `chest_pain`, `respiratory`, `trauma_injury`) | Derived from initial intake complaint string |
| `triage_pain` | `triage` | `float64` | Yes | Normalized triage pain score (0.0 to 10.0) | Normalized from text/numeric ratings |
| `triage_heartrate` | `triage` | `float64` | Yes | Initial triage heart rate (bpm) | Filtered for physiological limits |
| `triage_sbp`, `triage_dbp` | `triage` | `float64` | Yes | Initial systolic/diastolic BP (mmHg) | Filtered for bounds and $\text{SBP} \ge \text{DBP}$ |
| `triage_o2sat` | `triage` | `float64` | Yes | Initial oxygen saturation (%) | Plausibility filtered (50.0–100.0) |
| `triage_temperature` | `triage` | `float64` | Yes | Initial temperature (°F) | Plausibility filtered (85.0–108.0) |
| `triage_resprate` | `triage` | `float64` | Yes | Initial respiratory rate | Plausibility filtered (4.0–60.0) |
| `num_vital_measurements` | `vitalsign` | `int64` | Yes | Count of recorded vital sets $\le T_{\text{snapshot}}$ | Strictly ignores vitals with $charttime > T_{\text{snapshot}}$ |
| `last_heartrate` | `vitalsign` / `triage` | `float64` | Yes | Most recent heart rate before or at $T_{\text{snapshot}}$ | Falls back to triage heart rate if no longitudinal vital yet |
| `last_sbp`, `last_dbp` | `vitalsign` / `triage` | `float64` | Yes | Most recent blood pressure readings | Most recent reading strictly $\le T_{\text{snapshot}}$ |
| `last_o2sat` | `vitalsign` / `triage` | `float64` | Yes | Most recent oxygen saturation (%) | Most recent reading strictly $\le T_{\text{snapshot}}$ |
| `last_temperature` | `vitalsign` / `triage` | `float64` | Yes | Most recent temperature (°F) | Most recent reading strictly $\le T_{\text{snapshot}}$ |
| `last_resprate` | `vitalsign` / `triage` | `float64` | Yes | Most recent respiratory rate | Most recent reading strictly $\le T_{\text{snapshot}}$ |
| `mean_heartrate`, `min_heartrate`, `max_heartrate` | `vitalsign` | `float64` | Yes | Summary vital statistics up to $T_{\text{snapshot}}$ | Aggregated strictly over $charttime \le T_{\text{snapshot}}$ |
| `is_tachycardic` | derived | `int64`/`float` | Yes | Binary flag ($1$ if `last_heartrate` $> 100$) | Derived from snapshot-safe vital |
| `is_bradycardic` | derived | `int64`/`float` | Yes | Binary flag ($1$ if `last_heartrate` $< 60$) | Derived from snapshot-safe vital |
| `is_hypoxic` | derived | `int64`/`float` | Yes | Binary flag ($1$ if `last_o2sat` $< 92\%$) | Derived from snapshot-safe vital |
| `is_hypotensive` | derived | `int64`/`float` | Yes | Binary flag ($1$ if `last_sbp` $< 90\text{ mmHg}$) | Derived from snapshot-safe vital |
| `is_hypertensive_crisis` | derived | `int64`/`float` | Yes | Binary flag ($1$ if `last_sbp` $\ge 180\text{ mmHg}$) | Derived from snapshot-safe vital |
| `is_febrile` | derived | `int64`/`float` | Yes | Binary flag ($1$ if `last_temperature` $\ge 100.4^\circ\text{F}$) | Derived from snapshot-safe vital |

---

### 3.2 Temporal Features (`temporal_features.py`)

| Feature Name | Source | Type | Snapshot-Safe? | Preprocessing & Meaning | Leakage Considerations |
|---|---|---|:---:|---|---|
| `snapshot_hour` | derived | `int64` | Yes | Hour of day at snapshot (0–23) | Derived from $T_{\text{snapshot}}$ |
| `snapshot_day_of_week` | derived | `int64` | Yes | Day of week at snapshot (0=Mon, 6=Sun) | Derived from $T_{\text{snapshot}}$ |
| `snapshot_is_weekend` | derived | `int64` | Yes | $1$ if snapshot falls on Saturday or Sunday | Derived from $T_{\text{snapshot}}$ |
| `snapshot_month` | derived | `int64` | Yes | Calendar month (1–12) | Derived from $T_{\text{snapshot}}$ |
| `snapshot_hour_sin` | derived | `float64` | Yes | $\sin(2\pi \cdot \text{hour}/24.0)$ | Continuous cyclical hour representation |
| `snapshot_hour_cos` | derived | `float64` | Yes | $\cos(2\pi \cdot \text{hour}/24.0)$ | Continuous cyclical hour representation |
| `snapshot_shift` | derived | `str` | Yes | Nursing shift: `morning`, `evening`, `night` | Derived from $T_{\text{snapshot}}$ |
| `arrival_hour` | `edstays` | `int64` | Yes | Hour of patient presentation ($T_{\text{in}}$) | Historical registration time |
| `arrival_day_of_week` | `edstays` | `int64` | Yes | Day of week of presentation ($T_{\text{in}}$) | Historical registration time |
| `arrival_is_weekend` | `edstays` | `int64` | Yes | $1$ if presentation was on weekend | Historical registration time |
| `arrival_shift` | `edstays` | `str` | Yes | Shift bucket of patient presentation | Historical registration time |

---

### 3.3 Congestion Features (`congestion_features.py`)

| Feature Name | Source | Type | Snapshot-Safe? | Preprocessing & Meaning | Leakage Considerations |
|---|---|---|:---:|---|---|
| `active_patient_count` | `edstays` | `int64` | Yes | Number of patients active in ED at $T_{\text{snapshot}}$ ($T_{\text{in}} \le T_{\text{snapshot}} < T_{\text{out}}$) | Reflects real-time physical census; future timestamps are not exposed |
| `recent_arrivals_15m` | `edstays` | `int64` | Yes | Patient registrations in $(T - 15\text{m}, T]$ | Strictly counts arrivals $\le T_{\text{snapshot}}$ |
| `recent_arrivals_30m` | `edstays` | `int64` | Yes | Patient registrations in $(T - 30\text{m}, T]$ | Strictly counts arrivals $\le T_{\text{snapshot}}$ |
| `recent_arrivals_60m` | `edstays` | `int64` | Yes | Patient registrations in $(T - 60\text{m}, T]$ | Strictly counts arrivals $\le T_{\text{snapshot}}$ |
| `recent_arrivals_120m` | `edstays` | `int64` | Yes | Patient registrations in $(T - 120\text{m}, T]$ | Strictly counts arrivals $\le T_{\text{snapshot}}$ |
| `recent_departures_15m` | `edstays` | `int64` | Yes | Completed departures in $(T - 15\text{m}, T]$ | Strictly departures with $T_{\text{out}} \le T_{\text{snapshot}}$ |
| `recent_departures_30m` | `edstays` | `int64` | Yes | Completed departures in $(T - 30\text{m}, T]$ | Strictly departures with $T_{\text{out}} \le T_{\text{snapshot}}$ |
| `recent_departures_60m` | `edstays` | `int64` | Yes | Completed departures in $(T - 60\text{m}, T]$ | Strictly departures with $T_{\text{out}} \le T_{\text{snapshot}}$ |
| `recent_departures_120m` | `edstays` | `int64` | Yes | Completed departures in $(T - 120\text{m}, T]$ | Strictly departures with $T_{\text{out}} \le T_{\text{snapshot}}$ |
| `arrival_departure_ratio_60m` | derived | `float64` | Yes | $(arrivals_{60m} + 1.0) / (departures_{60m} + 1.0)$ | Smoothed arrival/discharge velocity ratio |
| `net_flow_60m` | derived | `int64` | Yes | $arrivals_{60m} - departures_{60m}$ | Queue accumulation rate over past hour |
| `high_acuity_active_count` | `triage` / `edstays` | `int64` | Yes | Number of active patients with ESI acuity $\le 2$ | Acuity pressure index at snapshot |
| `mean_active_acuity` | `triage` / `edstays` | `float64` | Yes | Mean ESI acuity across active patients | Overall acuity burden |

---

## 4. Dataset Observability Boundary

QueueMind strictly enforces a dataset observability boundary based on the empirical schema of MIMIC-IV-ED. Unobserved clinical workflow milestones are not modeled or claimed.

| Operational Quantity | Available in MIMIC-IV-ED? | Source Table / Fields | Timestamp Quality | Can Be Modeled? |
| :--- | :---: | :--- | :--- | :---: |
| ED Arrival / Registration Time | **Yes** | `edstays.intime` | Minute-level clinical recording | **Yes** (Arrival event) |
| ED Departure Time | **Yes** | `edstays.outtime` | Minute-level clinical recording | **Yes** (Target-side only) |
| Total ED Length of Stay | **Yes** | `outtime - intime` | Derived from arrival & departure | **Yes** (Target-side only) |
| Active Patient Census at $T$ | **Yes** | `edstays` ($T_{\text{in}} \le T < T_{\text{out}}$) | Reconstructed point-in-time state | **Yes** (Feature & Future Target) |
| Rolling Arrivals (15m, 30m, 60m, 120m) | **Yes** | `edstays.intime` | Exact event counts $\le T$ | **Yes** (Input Feature) |
| Rolling Departures (15m, 30m, 60m, 120m) | **Yes** | `edstays.outtime` | Exact completed departures $\le T$ | **Yes** (Input Feature) |
| Net Flow Velocity ($\Delta = \text{Arr} - \text{Dep}$) | **Yes** | Derived | Exact count differences $\le T$ | **Yes** (Input Feature) |
| Active Triage Acuity Composition | **Yes** | `edstays` + `triage.acuity` | Point-in-time join on active stays | **Yes** (Input Feature) |
| Door-to-Triage Wait Time | **No** | None | Not recorded in MIMIC-IV-ED (intime is intake) | **No** (Do not assume/model) |
| Door-to-Doctor Wait Time | **No** | None | No physician assignment timestamp exists | **No** (Do not assume/model) |
| Door-to-Bed Wait Time | **No** | None | No physical bed assignment timestamp exists | **No** (Do not assume/model) |
| Nurse Assignment Time | **No** | None | No nursing assignment timestamp exists | **No** (Do not assume/model) |
| Diagnostic Order Turnaround Time | **No** | None | Standalone MIMIC-IV-ED lacks lab turnaround | **No** (Do not assume/model) |
| Boarding Duration (Admit Decision to Depart) | **Partial/Proxy** | `edstays.disposition` | `disposition == 'ADMITTED'`, no boarding order stamp | Only as outcome label |

---

## 5. Department Time-Grid Features (`time_grid.py`)

All features are evaluated strictly at snapshot timestamp $T$ using information known $\le T$:

| Feature Name | Source | Type | Description |
|---|---|---|---|
| `current_active_census` | `edstays` | `int64` | Physical active census at $T$ ($T_{\text{in}} \le T < T_{\text{out}}$) |
| `recent_arrivals_15m` | `edstays` | `int64` | Registrations in $(T - 15\text{m}, T]$ |
| `recent_arrivals_30m` | `edstays` | `int64` | Registrations in $(T - 30\text{m}, T]$ |
| `recent_arrivals_60m` | `edstays` | `int64` | Registrations in $(T - 60\text{m}, T]$ |
| `recent_arrivals_120m` | `edstays` | `int64` | Registrations in $(T - 120\text{m}, T]$ |
| `recent_departures_15m` | `edstays` | `int64` | Completed discharges in $(T - 15\text{m}, T]$ |
| `recent_departures_30m` | `edstays` | `int64` | Completed discharges in $(T - 30\text{m}, T]$ |
| `recent_departures_60m` | `edstays` | `int64` | Completed discharges in $(T - 60\text{m}, T]$ |
| `recent_departures_120m` | `edstays` | `int64` | Completed discharges in $(T - 120\text{m}, T]$ |
| `net_flow_15m` | derived | `int64` | $arr_{15} - dep_{15}$ |
| `net_flow_30m` | derived | `int64` | $arr_{30} - dep_{30}$ |
| `net_flow_60m` | derived | `int64` | $arr_{60} - dep_{60}$ |
| `arrival_rate_per_hour_60m` | derived | `float64` | Hourly arrival velocity |
| `departure_rate_per_hour_60m` | derived | `float64` | Hourly departure throughput |
| `arrival_departure_ratio_60m` | derived | `float64` | $(arr_{60} + 1.0) / (dep_{60} + 1.0)$ |
| `high_acuity_active_count` | `triage` | `int64` | Number of active encounters with ESI $\le 2$ |
| `high_acuity_ratio` | derived | `float64` | Fraction of active encounters with ESI $\le 2$ |
| `mean_active_acuity` | `triage` | `float64` | Mean triage acuity of active encounters |
| `snapshot_hour` | derived | `int64` | Hour of snapshot (0–23) |
| `snapshot_day_of_week` | derived | `int64` | Day of week of snapshot (0=Mon, 6=Sun) |
| `snapshot_is_weekend` | derived | `int64` | Weekend indicator flag |
| `snapshot_hour_sin` | derived | `float64` | Cyclical sine hour |
| `snapshot_hour_cos` | derived | `float64` | Cyclical cosine hour |
| `snapshot_month` | derived | `int64` | Calendar month (1–12) |

---

## 6. Congestion Target Variables

| Target Name | Source | Type | Horizon | Segregation Rule |
|---|---|---|---|---|
| `target_census_30m` | `edstays` | `int64` | $+30\text{ minutes}$ | Physical census at $T + 30\text{m}$. Strictly prohibited from $X$. |
| `target_census_60m` | `edstays` | `int64` | $+60\text{ minutes}$ | Physical census at $T + 60\text{m}$. Strictly prohibited from $X$. |
| `target_census_120m` | `edstays` | `int64` | $+120\text{ minutes}$ | Physical census at $T + 120\text{m}$. Strictly prohibited from $X$. |

---

## 7. Operational Flow Indicators (`bottleneck_features.py`)

Mathematical queue dynamics signals (strictly non-clinical):

| Indicator | Condition | Operational Interpretation |
|---|---|---|
| `rising_census_velocity` | $net_{30m} \ge 3 \lor net_{60m} \ge 5$ | Rapid net patient accumulation in recent rolling intervals |
| `high_arrival_pressure` | $arr_{60m} \ge 8$ | Elevated intake pressure |
| `low_departure_throughput` | $dep_{60m} \le 2 \land census \ge 20$ | Discharge velocity stalled despite elevated active census |
| `sustained_positive_net_flow` | $net_{15m} > 0 \land net_{30m} > 0 \land net_{60m} > 0$ | Persistent queue expansion |
| `acuity_concentration` | $high\_acuity\_ratio \ge 0.40 \land census \ge 5$ | Large proportion of active beds occupied by critical patients |
