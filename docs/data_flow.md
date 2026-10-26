# Dual Forecasting Architecture

```mermaid
flowchart TD
    A[UCI minute-level household power] --> B[Hourly kWh cleaning]
    B --> C[Chronological historical series]
    C --> D[Pipeline A AutoGen ML agent]
    C --> E[Pipeline B AutoGen analog agent]
    D --> F[ML forecast tool: fit XGBoost XGBRegressor]
    E --> G[Analog forecast tool: same-hour median]
    C --> H[LlamaIndex daily summary embeddings]
    H --> I[Semantic retrieval RAG tool]
    I --> D
    I --> E
    F --> J[Validation tool]
    G --> K[Validation tool]
    J --> L[ML agent explanation and optimizer]
    K --> M[Agentic-only explanation and optimizer]
    L --> N[Streamlit comparison]
    M --> N
    C --> O[Chronological rolling backtest of numeric methods]
    O --> N
```

**Important:** The agentic-only approach has no trained ML forecaster, but uses a deterministic Python analog forecasting tool selected and invoked by AutoGen. LLMs are not asked to invent 24 numerical predictions. RAG retrieves historical summaries; it is not itself a forecasting model. Backtest metrics evaluate the numerical tools, not the quality of agent decisions.
