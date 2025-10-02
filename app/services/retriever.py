import numpy as np

class Retriever:
    def __init__(self, st, index, meta):
        self.st = st
        self.index = index
        self.meta = meta

    def embed_query(self, q: str) -> np.ndarray:
        try:
            v = self.st.encode([q], normalize_embeddings=True, prompt_name="query")[0]
        except TypeError:
            v = self.st.encode([f"query: {q}"], normalize_embeddings=True)[0]
        return v.astype("float32")
    
    def search(self, q: str, topk = 5):
        qv = self.embed_query(q)
        D, I = self.index.search(qv[None, :], topk)
        rows = []
        for s, idx in zip(D[0], I[0]):
            m = self.meta.iloc[int(idx)]
            rows.append({
                "row_id": int(idx),
                "score": float(s),
                "title": m["title"],
                "doc_id": m["doc_id"],
                "chunk_id": int(m["chunk_id"])
            })

        return rows
