from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import faiss

def init_models(app):
    # Load Path
    META_PATH = "/data/meta.parquet"
    INDEX_PATH = "/data/index.faiss"
    CHUNKS_PATH = "/data/chunks.parquet"

    # Load Embedding
    st = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    index = faiss.read_index(INDEX_PATH)
    meta = pd.read_parquet(META_PATH)
    chunks_df = pd.read_parquet(CHUNKS_PATH)

    # Load Qwen
    GEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    app.state.st = st
    app.state.index = index
    app.state.meta = meta
    app.state.chunks_df = chunks_df
    app.state.tok = tok
    app.state.model = model

    print("Model Initialized")