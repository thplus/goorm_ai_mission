from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

def init_models(app):
    # Load Path
    BASE = Path(__file__).resolve().parents[1]
    DATA = BASE / "data"

    META_PATH = DATA / "meta.parquet"
    INDEX_PATH = DATA / "index.faiss"
    CHUNKS_PATH = DATA / "chunks.parquet"

    # Load Embedding
    st = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)
    chunks_df = pd.read_parquet(CHUNKS_PATH)

    # Load Qwen
    GEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(GEN_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
    )

    app.state.st = st
    app.state.index = index
    app.state.meta = meta
    app.state.chunks_df = chunks_df
    app.state.tok = tok
    app.state.model = model

    print("Model Initialized")