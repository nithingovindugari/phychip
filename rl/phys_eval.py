"""Physics held-out eval: greedy answer, parse 'Answer: <number>', correct if within max(rel_tol, 2%). usage: phys_eval.py MODEL TASKS OUT.json"""
import json, re, sys
if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    model, tasks_path, out = sys.argv[1:4]
    SYS = "You are an expert in semiconductor physics and electrical engineering. Solve the problem step by step, concisely, and end with a final line 'Answer: <number> <unit>'."
    tasks = [json.loads(l) for l in open(tasks_path)]
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    prompts = []
    for t in tasks:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": t["question"]}]
        try: prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False))
        except TypeError: prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1500))
    ok = 0; rows = []
    for t, o in zip(tasks, outs):
        txt = o.outputs[0].text; m = re.findall(r"Answer:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", txt)
        v = float(m[-1]) if m else None
        good = v is not None and abs(v - t["answer"]) <= max(t["rel_tol"], 0.02) * max(abs(t["answer"]), 1e-30)
        ok += good; rows.append({"id": t["id"], "pred": v, "answer": t["answer"], "correct": good})
    json.dump({"model": model, "n": len(tasks), "correct": ok, "acc": ok / len(tasks), "rows": rows}, open(out, "w"))
    print("PHYS", model, ok, "/", len(tasks))
