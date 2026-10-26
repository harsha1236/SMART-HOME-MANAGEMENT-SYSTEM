# Smart Home Energy Forecasting — XGBoost + two AutoGen workflows

This version implements **two separate tool-using AutoGen forecasting agents**:

- **Pipeline A: ML + Agentic AI.** The AutoGen agent MUST call the ML training + 24h prediction tool, validate the result with another tool, and query LlamaIndex RAG. An Ollama-powered optimization agent then makes recommendations.
- **Pipeline B: Agentic AI only.** A different AutoGen agent MUST call a non-ML historical-analog numerical forecast tool, validate it, and query LlamaIndex RAG. Its own optimization agent makes recommendations.

Both branches use **Ollama** locally, and the dashboard compares their 24-hour outputs. An absent tool call triggers an explicit error rather than silently showing a precomputed result as agent-generated.

## Setup in VS Code (macOS)

1. Install **Python 3.11** and [Ollama](https://ollama.com/download).
2. Open this folder in VS Code and open its terminal.
3. Run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2
streamlit run app.py
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2
streamlit run app.py
```

If Ollama is not already running, start it in another terminal with `ollama serve`. Open the Streamlit URL (usually http://localhost:8501). The first launch downloads the UCI dataset; the first RAG run downloads the local Hugging Face embedding model.

Click **Run BOTH real agent-driven pipelines**. This invokes two independent AutoGen agents, each with its own numeric forecasting tool, RAG retrieval tool and inspection tool. Inspect **Verified agent tool calls** for evidence of execution.

## Data and evaluation

- Public dataset: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), 2006–2010, CC BY 4.0.
- Minute kW readings converted to hourly kWh; select contiguous historical data.
- ML: XGBoost XGBRegressor, recursive 24h prediction, calendar/lag features.
- Agentic-only: no trained ML forecasting model; median of historical same-hour analogs at 1, 2, 7, 14 days.
- Backtesting: final 168 hours, rolling 24-hour origins, MAE/RMSE for underlying numeric methods plus seasonal naive.
- **Evaluation does not run the Ollama agents at every historical backtest origin.** It evaluates the numerical policies those agents call, so don't interpret the scores as LLM decision-quality metrics.
- The RAG index is in-memory and built for explanation of the final historical forecast. It is not used to access future test labels during the numerical backtest.

## Limitations

- This is a historical demonstration, not a live forecast for your home today.
- The LLM is an orchestrator and explainer; it does not independently generate numerical readings. The second branch is genuinely agent-orchestrated but uses a deterministic analog forecasting tool.
- Ollama model must support tool calls; tool selection may vary. If required calls aren't made, the app reports a failure.
- No actual appliance switching, tariff optimization, or verified cost savings.
- Numerical forecasts should not be confused with guaranteed future consumption.
- End-to-end execution with downloaded data and a running Ollama instance must be verified on your machine.

## Files

- `app.py`: Streamlit UI and side-by-side agent execution
- `src/pipeline.py`: public data, ML and analog tools, numerical backtest
- `src/rag.py`: LlamaIndex vector retrieval over daily summaries
- `src/agents.py`: independent AutoGen tool-using workflows with Ollama
- `docs/data_flow.md`: architecture diagram

## XGBoost settings

Pipeline A uses `xgboost.XGBRegressor` with 250 estimators, max depth 5, learning rate 0.05, histogram tree method, 0.9 subsampling, and fixed seed 42. Both `evaluate()` and the agent-called `fit_ml_model()` use the same model family and settings. Pipeline B is unchanged and has **no trained ML model**.


## Reliable orchestration update
The Python orchestrator now guarantees forecast, validation and LlamaIndex retrieval tool execution before invoking AutoGen analyst and optimizer agents. Tool logs label calls as `executed_by: orchestrator`; the LLM is not claimed to have autonomously selected these tools. Numerical backtest scores still describe the underlying forecasting methods, not the agent explanations.
