from fastapi import FastAPI
from startup import init_models
from pydantic import BaseModel
from services.retriever import Retriever
from services.contextbuilder import ContextBuilder
from services.generate import Generator

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_models(app)

class QwenRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QwenRequest):
    topk, cutoff, k_ctx, min_score = 5, 0.36, 5, 0.0

    retriver = Retriever(app.state.st, app.state.index, app.state.meta) #st, index, meta
    ctxb = ContextBuilder(app.state.chunks_df) #chunks_df
    generator = Generator(app.state.tok, app.state.model) #tok, model

    question = req.question

    hits = retriver.search(req.question, topk=topk)
    ctx_blocks, sources, used_chunks, top1 = ctxb.build_context(hits, k_ctx=k_ctx, min_score=min_score)

    if (not hits) or (top1 < cutoff) or (not ctx_blocks):
        return {
            "retrieved_document_id": 0,
            "retrieved_document": "",
            "question": question,
            "answers": "제공된 문서에 정보가 없습니다."
        }
    
    answer = generator.generate_answer(question, ctx_blocks)

    return {
        "retrieved_document_id": int(sources[0].split(":")[0]),
        "retrieved_document": used_chunks[0],
        "question": question,
        "answers": answer
    }