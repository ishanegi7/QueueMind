# QueueMind Operational Simulation & What-If Engine Guide

## 1. Overview & Purpose

The **Operational Simulation & What-If Engine** translates QueueMind's machine learning predictions into actionable administrative intelligence for Emergency Department (ED) charge nurses, flow coordinators, and hospital operations managers.

### 1.1 Non-Clinical Operational Scope
- **Domain**: Queue dynamics, bed-occupancy stress, presentation surges, and discharge velocity.
- **Explicit Non-Goal**: Clinical decision support, patient diagnosis, triage downgrading/upgrading, or medical intervention selection.
- **Architectural Separation**: Observed historical data, ML predictions, deterministic scenario assumptions, and simulated counterfactuals are strictly segregated in memory and outputs.

---

## 2. Queue Health Score (0–100)

The **Queue Health Score** is a composite operational index quantifying department-level queue strain on a standardized 0–100 scale.

### 2.1 Component Mathematical Formulations

The score aggregates three distinct operational pressure vectors:

1. **Congestion Pressure ($P_{\text{congestion}}$)**:
   $$\text{Ratio} = \frac{\text{Active Census}}{\text{Capacity Reference}}$$
   $$P_{\text{congestion}} = \min\left(100.0, \max\left(0.0, \text{Ratio} \times 100.0\right)\right)$$
   *Default Capacity Reference*: 50 beds (configurable abstract operational parameter).

2. **Arrival Velocity Pressure ($P_{\text{arrivals}}$)**:
   $$\text{Ratio} = \frac{\text{Recent Arrivals (60m)}}{\text{Arrival Rate Reference}}$$
   $$P_{\text{arrivals}} = \min\left(100.0, \max\left(0.0, \text{Ratio} \times 100.0\right)\right)$$
   *Default Arrival Rate Reference*: 12 presentations / hour.

3. **High-Acuity Workload Pressure ($P_{\text{acuity}}$)**:
   $$\text{Ratio} = \frac{\text{High Acuity Ratio (ESI } \le 2\text{)}}{\text{High Acuity Reference}}$$
   $$P_{\text{acuity}} = \min\left(100.0, \max\left(0.0, \text{Ratio} \times 100.0\right)\right)$$
   *Default High Acuity Reference*: 0.30 (30% of active census).

### 2.2 Composite Weighting

The raw score is computed as a convex combination of normalized component pressures:
$$\text{Raw Score} = w_{\text{congestion}} P_{\text{congestion}} + w_{\text{arrivals}} P_{\text{arrivals}} + w_{\text{acuity}} P_{\text{acuity}}$$
$$\text{Queue Health Score} = \text{clip}\left(\text{round}(\text{Raw Score}, 1), 0.0, 100.0\right)$$

**Constraint**: $\sum w_i = 1.0$, with $w_i > 0$.
- Default: $w_{\text{congestion}} = 0.50$, $w_{\text{arrivals}} = 0.30$, $w_{\text{acuity}} = 0.20$.

### 2.3 Operational Health States

| State | Score Range | Operational Meaning | Suggested Administrative Posture |
|:---|:---:|:---|:---|
| **`HEALTHY`** | $0.0 \le \text{Score} \le 30.0$ | Normal operational volume; adequate buffer capacity | Standard staffing and discharge scheduling |
| **`MODERATE`** | $30.0 < \text{Score} \le 60.0$ | Typical operational load; manageable intake rate | Active monitoring; standard patient turnover |
| **`BUSY`** | $60.0 < \text{Score} \le 80.0$ | Elevated queue strain; approaching bed saturation | Expedite pending inpatient bed transfers |
| **`CRITICAL`** | $80.0 < \text{Score} \le 100.0$ | Severe bottlenecking; capacity deficit or intake shock | Flow escalation; trigger surge response protocols |

### 2.4 Dominant Factor Attribution
The system identifies which component contributes the greatest weighted pressure:
$$\text{Dominant Factor} = \arg\max_{k \in \{\text{congestion}, \text{arrivals}, \text{acuity}\}} \left(w_k \times P_k\right)$$
Provides clear administrative explanations (e.g. *"Primary operational pressure driver is rapid presentation velocity"*).

---

## 3. What-If Scenario Simulation Engine

The simulation engine models how operational perturbations propagate through department flow over a forward planning horizon.

### 3.1 Discrete-Time Flow Conservation
All simulations adhere to the physical conservation of patient volume:
$$C(t + \Delta t) = \max\left(0.0, C(t) + \text{Arrivals}(t) - \text{Departures}(t)\right)$$
Clamping at zero guarantees active census cannot become negative even under aggressive discharge acceleration.

### 3.2 Implemented Counterfactual Scenarios

#### Scenario A: Discharge Acceleration (+X%)
- **Mechanism**: Scales departure velocity by $(1 + \alpha)$, where $\alpha \ge 0$ (e.g. $\alpha = 0.20$ for $+20\%$ throughput).
- **Assumptions**: Arrival trajectory remains invariant; discharge acceleration is applied uniformly across intervals.
- **Outputs**: Simulated census trajectory, peak census reduction, final census delta, updated Queue Health Score.

#### Scenario B: Capacity Constraint Reduction
- **Mechanism**: Evaluates flow under a lowered operational bed ceiling (e.g. closing an 8-bed pod for staffing shortages).
- **Assumptions**: Physical census follows baseline flow, but department capacity reference is tightened to the reduced threshold.
- **Outputs**: Peak bed overflow ($\max(0.0, \text{Peak Census} - \text{Capacity})$), elevated Queue Health Score, updated stability state.

#### Scenario C: Arrival Surge Shocks (+N presentations over M intervals)
- **Mechanism**: Injects $N$ additional presentations uniformly across the first $M$ time steps ($N / M$ additional arrivals per step).
- **Assumptions**: Baseline departure rate remains constant (no immediate automatic staffing escalation).
- **Outputs**: Surge absorption trajectory, post-surge recovery dynamics, peak census elevation, surge queue health impact.

---

## 4. Operational Queue Stability Evaluation

Queue stability categorizes department momentum over the simulation horizon:

$$\text{Net Flow} = \sum_{t} \text{Arrivals}(t) - \sum_{t} \text{Departures}(t)$$
$$\Delta \text{Census} = C_{\text{final}} - C_{\text{initial}}$$

| Stability Category | Mathematical Criteria | Operational Interpretation |
|:---|:---|:---|
| **`STABLE`** | $\text{Net Flow} \le 0.0$ AND $\Delta \text{Census} \le 0.0$ | Intake is balanced or exceeded by discharges; queue is stable or clearing. |
| **`STRAINED`** | $\text{Net Flow} > 0.0$ AND $\text{Net Flow} \le 10.0$ AND $C_{\text{final}} \le 1.5 \times C_{\text{initial}}$ | Intake outpaces discharges but accumulation remains bounded. |
| **`UNSTABLE`** | $\text{Net Flow} > 10.0$ OR $C_{\text{final}} > 1.5 \times C_{\text{initial}}$ | Rapid runaway accumulation; queue growth is unsustainable without intervention. |

---

## 5. Non-Causal Semantics & Limitations

### 5.1 The Causal Inference Boundary
> [!IMPORTANT]
> **MIMIC-IV-ED contains observational, retrospective electronic health record data without interventional counterfactuals.**
> It records what occurred historically; it does not record what would have happened if administrative actions were altered.

### 5.2 Waiting-Time Impact Handling
Any claim that "+20% discharge acceleration will reduce Patient X's wait by Y minutes" is scientifically invalid without unmeasured confounder control and counterfactual timestamps.
To prevent misleading claims, QueueMind:
1. Returns an explicit `waiting_time_impact` status payload:
   ```json
   {
     "status": "unavailable",
     "reason": "Current dataset lacks interventional counterfactual timestamps. Direct causal waiting-time reduction claims are scientifically unsupported.",
     "operational_proxy": "Active census changes reflect aggregate department bed-load adjustments, not individual patient wait time guarantees."
   }
   ```
2. Restricts scenario metrics to **aggregate departmental bed-load** (active census headcounts and queue stability), which obey conservation of mass.

---

## 6. Programmatic Usage Examples

### 6.1 Computing Queue Health Score
```python
from queuemind.queue_health import calculate_queue_health_score, QueueHealthConfig

# Evaluate current state
health = calculate_queue_health_score(
    active_census=45.0,
    recent_arrivals_60m=10.0,
    high_acuity_ratio=0.25,
)

print(f"Score: {health['score']}/100")
print(f"State: {health['state']}")
print(f"Dominant Factor: {health['dominant_factor']}")
print(f"Summary: {health['summary']}")
```

### 6.2 Running a What-If Discharge Acceleration Simulation
```python
import pandas as pd
from queuemind.simulation import BaselineTrajectory, simulate_discharge_acceleration

# Define 1-hour baseline (4 x 15-minute intervals)
start = pd.Timestamp.now()
time_steps = [start + pd.Timedelta(minutes=15 * i) for i in range(5)]

baseline = BaselineTrajectory(
    time_steps=time_steps,
    initial_census=42.0,
    arrivals=[4.0, 5.0, 4.0, 3.0],
    departures=[3.0, 4.0, 3.0, 4.0],
    high_acuity_ratio=0.20,
)

# Simulate +25% discharge throughput
result = simulate_discharge_acceleration(baseline, acceleration_rate=0.25)

print(f"Scenario: {result.scenario_name}")
print(f"Peak Census: {result.peak_baseline_census} -> {result.peak_simulated_census}")
print(f"Peak Reduction: {result.peak_delta} patients")
print(f"Stability: {result.stability}")
print(f"Waiting Time Status: {result.waiting_time_impact['status']}")
```
