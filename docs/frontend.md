# Frontend Documentation

QueueMind's frontend is a Next.js (App Router) operational dashboard providing situational awareness via Queue Health scores, multi-horizon congestion forecasts, patient-flow journey predictions, and what-if simulations.

## Tech Stack
- Next.js (App Router) v14/15
- React 19
- TypeScript (Strict Mode)
- Tailwind CSS v4
- Recharts (Visualizations)

## Architecture

- **`app/`**: Top-level routes (`/`, `/analytics`, `/simulation`).
- **`components/`**: Reusable views divided by domain.
  - `common/`: Badges, layouts, loading/error states.
  - `dashboard/`: Queue health hero, flow KPIs, bottleneck panel.
  - `patient-flow/`: Patient prediction form, SHAP explanation charts.
  - `simulation/`: Scenario parameter inputs, comparison table.
  - `charts/`: Recharts wrappers (AreaChart, Gauge, LineChart).
- **`lib/types.ts`**: Frontend TypeScript definitions exactly mirroring the FastAPI Pydantic v2 schemas.
- **`lib/api.ts`**: Typed `fetch` wrapper. Correctly throws `ApiConnectionError`, `ApiValidationError`, and `ApiServiceError`.

## API Integration & Demo Strategy

The frontend strictly enforces a **no-fabricated-data** rule. 
- If the API server is down, the frontend shows an `ApiConnectionError` stating the API is unreachable.
- If the API server is up, but the ML models are not mounted (HTTP 503), the frontend shows an `ApiServiceError` stating models are unavailable.
- There is no bundle containing MIMIC data and no fake predictions are embedded in the code.

## Running Locally

1. **Start Backend**
   ```bash
   cd QueueMind
   source venv/Scripts/activate
   uvicorn api.main:app --reload
   ```

2. **Start Frontend**
   ```bash
   cd QueueMind/frontend
   npm install
   npm run dev
   ```

3. Navigate to `http://localhost:3000`. Ensure `.env` is configured properly for CORS (`ALLOWED_ORIGINS`).
