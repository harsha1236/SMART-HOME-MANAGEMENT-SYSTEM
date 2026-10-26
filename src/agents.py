"""Reliable dual forecasting workflows with mandatory tool execution and AutoGen analysis.

The Python orchestrator guarantees forecasting, inspection and RAG retrieval.
AutoGen/Ollama handles explanation and recommendations; tool logs distinguish
orchestrator calls from agent calls. No unverified claim of autonomous tool selection.
"""
import asyncio
import json
import numpy as np
import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from src.pipeline import fit_ml_model, recursive_forecast, rag_numeric_forecast
from src.rag import build_retriever, retrieve_context


def _readable_forecast(values, origin):
    idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=len(values), freq='h')
    return [{'timestamp': str(t), 'kwh': round(float(v), 5)} for t, v in zip(idx, values)]


async def _run_branch(df, branch, model_name):
    series = df['kwh'].asfreq('h')
    if series.isna().any():
        raise ValueError('Non-contiguous history supplied to agent')
    history = series.copy()
    events = []
    predictions = {}

    def retrieve_energy_context(question: str) -> str:
        """Retrieve historical household energy summaries via LlamaIndex RAG."""
        retriever = build_retriever(df)
        result = retrieve_context(retriever, question)
        events.append({'tool': 'retrieve_energy_context', 'question': question, 'executed_by': 'orchestrator'})
        return result[:6500]

    def forecast_with_ml(horizon: int = 24) -> str:
        """Train XGBoost on historical data and predict next 24 hours."""
        if horizon != 24:
            raise ValueError('This demo supports exactly 24 hours')
        model = fit_ml_model(history)
        values = np.asarray(recursive_forecast(model, history, horizon), dtype=float)
        predictions['forecast'] = values
        events.append({'tool': 'forecast_with_ml', 'hours': horizon, 'executed_by': 'orchestrator'})
        return json.dumps({'method': 'XGBoost + Agentic AI', 'hours': _readable_forecast(values, history.index[-1]), 'total_kwh': round(float(values.sum()), 4)})

    def forecast_with_analogs(horizon: int = 24) -> str:
        """Predict using historical same-hour analogs; no trained ML model."""
        if horizon != 24:
            raise ValueError('This demo supports exactly 24 hours')
        values = np.asarray(rag_numeric_forecast(history, horizon), dtype=float)
        predictions['forecast'] = values
        events.append({'tool': 'forecast_with_analogs', 'hours': horizon, 'executed_by': 'orchestrator'})
        return json.dumps({'method': 'Agentic AI with non-ML analog tool', 'hours': _readable_forecast(values, history.index[-1]), 'total_kwh': round(float(values.sum()), 4)})

    def inspect_forecast() -> str:
        """Validate forecast values and summarize peak and total consumption."""
        if 'forecast' not in predictions:
            raise RuntimeError('No forecast exists')
        arr = predictions['forecast']
        events.append({'tool': 'inspect_forecast', 'executed_by': 'orchestrator'})
        info = {'count': len(arr), 'all_finite': bool(np.isfinite(arr).all()),
                'all_nonnegative': bool((arr >= 0).all()),
                'total_kwh': round(float(arr.sum()), 3),
                'peak_hour_ahead': int(arr.argmax()) + 1,
                'peak_hour_kwh': round(float(arr.max()), 3)}
        if info['count'] != 24 or not info['all_finite'] or not info['all_nonnegative']:
            raise ValueError(f'Invalid forecast: {info}')
        return json.dumps(info)

    # Deterministic execution: no dependence on the LLM choosing to call every tool.
    forecast_tool = forecast_with_ml if branch == 'ml' else forecast_with_analogs
    forecast_tool(horizon=24)
    validation = inspect_forecast()
    context = retrieve_energy_context('What historical daily energy consumption and peak-hour patterns are relevant to a 24-hour household forecast?')
    forecast_summary = {'branch': branch, 'validation': json.loads(validation),
                        'hourly_forecast': _readable_forecast(predictions['forecast'], history.index[-1])}

    client = OllamaChatCompletionClient(model=model_name)
    try:
        analyst = AssistantAgent(
            name=f'{branch}_analyst', model_client=client,
            system_message=('You are an energy forecast analyst. Explain the provided tool-computed '
                            '24-hour forecast and historical RAG context. Never invent values. '
                            'Mention uncertainty and do not claim RAG calculated the numbers.'))
        analysis = await analyst.run(task=(
            'Analyze this VERIFIED numerical forecast and retrieved history. '
            'All mandatory tools have already executed successfully. '
            f'Forecast and validation: {json.dumps(forecast_summary)}\n'
            f'Historical RAG context:\n{context}'))
        explanation = str(analysis.messages[-1].content)
        optimizer = AssistantAgent(
            name=f'{branch}_optimizer', model_client=client,
            system_message=('Give exactly three conditional, practical energy-use suggestions. '
                            'Do not invent savings estimates or claim appliance control.'))
        advice = await optimizer.run(task=f'Based on this forecast analysis, recommend 3 safe actions:\n{explanation}')
        return {'branch': branch, 'forecast': predictions['forecast'].tolist(),
                'explanation': explanation, 'recommendations': str(advice.messages[-1].content),
                'tool_log': events}
    finally:
        await client.close()


async def _run_both(df, model_name):
    a = await _run_branch(df, 'ml', model_name)
    b = await _run_branch(df, 'agentic', model_name)
    return {'ml': a, 'agentic': b}


def run_both_workflows(df, model_name='llama3.2'):
    return asyncio.run(_run_both(df, model_name))
