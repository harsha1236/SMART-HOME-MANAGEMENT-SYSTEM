import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def build_retriever(df):
    # Indexed documents are training-history summaries only; no test labels leak into RAG.
    recent=df['kwh'].iloc[-24*90:]
    documents=[]
    for day, part in recent.groupby(recent.index.date):
        if len(part)<20:continue
        peak=part.idxmax()
        documents.append(Document(text=(f'Date {day}; consumption {part.sum():.2f} kWh; '
            f'average hourly {part.mean():.2f} kWh; peak {part.max():.2f} kWh at {peak.hour}:00; '
            f'weekday {pd.Timestamp(day).day_name()}.'), metadata={'date':str(day)}))
    documents.append(Document(text='Forecasting guideline: kWh is energy, kW is power. '
        'Shift flexible loads from peak hours when convenient. Forecasts are estimates, not guaranteed savings.'))
    Settings.embed_model=HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
    index=VectorStoreIndex.from_documents(documents)
    return index.as_retriever(similarity_top_k=5)


def retrieve_context(retriever, query):
    return '\n'.join(n.node.get_content() for n in retriever.retrieve(query))
