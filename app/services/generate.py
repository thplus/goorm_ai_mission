import torch

class Generator:
    def __init__(self, tok, model):
        self.SYSTEM = (
            "당신은 한국어 RAG 어시스턴트입니다. 제공된 근거(context)만 사용해 사실적으로 답하세요. "
            "근거에 없으면 '제공된 문서에 정보가 없습니다.'라고 한 줄로만 답하세요."
        )
        self.tok = tok
        self.model = model

    def build_prompt(self, question, ctx_blocks):
        user = (
            "[문서들]\n" + "\n".join(f"- {b}" for b in ctx_blocks) + "\n\n"
            "[질문]\n" + question + "\n\n"
            "[출력 지침]\n- 한 줄로만 답하세요. 추가 설명/머리말/코드블록 금지."
        )
        return self.tok.apply_chat_template(
            [{"role":"system","content":self.SYSTEM},
             {"role":"user","content":user}],
            tokenize=False, add_generation_prompt=True
        )
    
    def generate_answer(self, question, ctx_blocks, temperature=0.2, max_new_tokens=128):
        prompt = self.build_prompt(question, ctx_blocks)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=temperature, top_p=0.9,
                max_new_tokens=max_new_tokens, eos_token_id=self.tok.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return text.splitlines()[0].strip() if text else "제공된 문서에 정보가 없습니다."