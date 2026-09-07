# Cloud Architecture

The following diagram illustrates the deployment foundation of QueueMind, distinguishing between the **Current (Implemented)** components and the **Future (Planned)** Google Cloud integrations.

```mermaid
flowchart TD
    %% Define Styles
    classDef current fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    classDef future fill:#0f172a,stroke:#fbbf24,stroke-width:2px,stroke-dasharray: 5 5,color:#94a3b8
    classDef user fill:#334155,stroke:#475569,stroke-width:2px,color:#f8fafc

    %% User Interaction
    Client[User / Browser]:::user

    %% Current Components
    subgraph Frontend [Presentation Layer]
        NextJS[Next.js Frontend\n(Local / Vercel)]:::current
    end

    subgraph Backend [API Layer]
        FastAPI[FastAPI Backend\nDocker Container]:::current
        Models[Local Model Artifacts\n(.joblib)]:::current
    end

    %% Planned Components
    subgraph GoogleCloud [Planned Google Cloud Architecture]
        CloudRun[Cloud Run\n(Serverless API hosting)]:::future
        GCS[Cloud Storage\n(Model Artifacts)]:::future
        BigQuery[BigQuery\n(Historical ED Data)]:::future
        Firebase[Firebase\n(Auth & State)]:::future
        Gemini[Vertex AI / Gemini\n(LLM Analysis)]:::future
    end

    %% Current Connections
    Client -- HTTP/UI --> NextJS
    NextJS -- HTTPS / JSON --> FastAPI
    FastAPI -- Loads --> Models

    %% Future Connections
    FastAPI -. Deploy to .-> CloudRun
    CloudRun -. Downloads on boot .-> GCS
    CloudRun -. Queries .-> BigQuery
    NextJS -. Authenticates .-> Firebase
    CloudRun -. Prompts .-> Gemini
```

### Component Status

| Component | Status | Description |
| :--- | :--- | :--- |
| **Next.js Frontend** | Current | Fully implemented React application querying the FastAPI backend. |
| **FastAPI Backend** | Current | Container-ready REST API implementing patient flow and congestion logic. |
| **Docker Foundation** | Current | `Dockerfile` and `docker-compose.yml` configured for deployment. |
| **Cloud Run** | Planned | Target hosting environment for the FastAPI backend. |
| **Cloud Storage** | Planned | Future repository for trained model artifacts. |
| **BigQuery / Firebase** | Planned | Future infrastructure for historical analytics and user state. |
| **Gemini / Vertex AI** | Planned | Future integration for unstructured operational insights. |
