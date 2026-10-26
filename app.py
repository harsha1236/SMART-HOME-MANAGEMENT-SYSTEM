import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline import prepare_data, evaluate
from src.agents import run_both_workflows

st.set_page_config(page_title='Dual Pipeline Smart Home Energy Forecasting', layout='wide')
st.title('⚡ Smart Home Energy Forecasting — Two Agentic Pipelines')
st.caption('Pipeline A: ML + Agentic AI | Pipeline B: Agentic AI without trained ML forecaster')
st.warning('Historical UCI 2006–2010 data: this predicts the 24 hours after the last dataset timestamp, NOT tomorrow in real time.')

@st.cache_data(show_spinner='Downloading/processing public UCI dataset...')
def load():
    return prepare_data()

@st.cache_data(show_spinner='Backtesting numerical forecasting policies...')
def evaluate_cached():
    return evaluate(load())

try:
    df = load()
    scores, comparison, baseline_future = evaluate_cached()
except Exception as exc:
    st.error(f'Unable to load/evaluate dataset: {exc}')
    st.stop()

st.sidebar.metric('Hourly readings', f'{len(df):,}')
st.sidebar.caption(f'Data through {df.index[-1]}')
model_name = st.sidebar.text_input('Local Ollama model (tool calling required)', 'llama3.2')
st.subheader('Historical energy (last 14 days)')
st.line_chart(df['kwh'].tail(24*14))
st.subheader('Numerical backtest: rolling 24-hour origins')
st.dataframe(scores.round(4), hide_index=True, use_container_width=True)
st.caption('These MAE/RMSE scores evaluate the underlying forecasting methods, not Ollama tool-choice reliability or explanation quality.')
st.line_chart(comparison.tail(72))
st.subheader('Reproducible numerical baseline forecasts')
st.line_chart(baseline_future)
st.info('Click below to execute TWO independent AutoGen forecasting agents. Each executes its forecasting, validation, and LlamaIndex RAG tools in a guaranteed order before AutoGen/Ollama explains the results.')
if st.button('Run BOTH real agent-driven pipelines', type='primary'):
    try:
        with st.spinner('Building LlamaIndex RAG and running AutoGen agents through Ollama (may take a few minutes)...'):
            st.session_state['dual_result'] = run_both_workflows(df, model_name)
    except Exception as exc:
        st.error(f'Agent workflow did not complete: {exc}')
        st.info('Confirm Ollama is running, the model is pulled, and the model supports tool calling. No fake agent results are substituted.')

if 'dual_result' in st.session_state:
    results = st.session_state['dual_result']
    idx = baseline_future.index
    agent_forecasts = pd.DataFrame({
        'ML + Agentic AI': results['ml']['forecast'],
        'Only Agentic AI': results['agentic']['forecast'],
    }, index=idx)
    st.subheader('Agent-supported 24-hour forecasts — side by side')
    st.line_chart(agent_forecasts)
    a, b = st.columns(2)
    for col, key, title in [(a, 'ml', 'Pipeline A — ML + Agentic AI'), (b, 'agentic', 'Pipeline B — Only Agentic AI')]:
        with col:
            st.markdown(f'### {title}')
            st.metric('Forecast total', f"{sum(results[key]['forecast']):.2f} kWh")
            st.markdown('**Agent explanation**')
            st.write(results[key]['explanation'])
            st.markdown('**Optimization agent recommendations**')
            st.write(results[key]['recommendations'])
            with st.expander('Verified pipeline tool executions'):
                st.json(results[key]['tool_log'])
    st.download_button('Download BOTH agent forecasts (CSV)',
                       agent_forecasts.to_csv(index_label='timestamp'),
                       file_name='agentic_comparison.csv', mime='text/csv')
else:
    st.caption('Agent-generated forecasts appear here after both workflows finish. Numerical baselines above can be viewed without Ollama.')
