class ContextBuilder:
    def __init__(self, chunks_df):
        self.chunks = chunks_df
    
    def get_chunk_text(self, doc_id, chunk_id, prefer="context"):
        row = self.chunks[(self.chunks["doc_id"]==doc_id) & (self.chunks["chunk_id"]==int(chunk_id))]
        if row.empty: 
            return ""
        
        for col in (prefer, "norm_text", "context", "text"):
            if col in row.columns:
                val = row.iloc[0][col]
                if val is not None:
                    return str(val)
                
        return ""
    
    def build_context(self, hits, k_ctx=5, min_score=0.0, max_chars=500):
        blocks, sources, used = [], [], []

        if not hits:
            return blocks, sources, used, 0.0
        
        top1_score = hits[0].get("score", 0.0)

        for h in hits[:k_ctx]:
            if h.get("score", 1.0) < min_score:
                continue
            chunk = self.get_chunk_text(h["doc_id"], h["chunk_id"], prefer="context")
            snippet = (chunk or "").replace("\n", " ").strip()[:max_chars]
            blocks.append(snippet)
            sources.append(f'{h.get("row_id",0)}: {h.get("title","")}#{h.get("chunk_id",0)}')
            used.append(chunk or "")

        seen, uniq = set(), []
        for s in sources:
            if s not in seen:
                seen.add(s); uniq.append(s)

        return blocks, uniq, used, top1_score