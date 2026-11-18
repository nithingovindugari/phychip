"""Fast batched sampling with vLLM (offline engine). Same I/O contract as gen.py.

python vgen.py --model DIR --tasks tasks.jsonl --out out.jsonl --k 8 --temperature 0.8 [--greedy]
"""
import argparse, json, time
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--also-greedy", action="store_true", help="write OUT.greedy.jsonl too, one engine load")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks)]
    import os
    cfg_path = os.path.join(args.model, "config.json")
    is_vl = os.path.exists(cfg_path) and "vision_config" in json.load(open(cfg_path))
    mm = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if is_vl else {}
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.gpu_frac,
              seed=args.seed, trust_remote_code=True, **mm)
    tok = llm.get_tokenizer()


    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


    prompts = [render(t["messages"]) for t in tasks]


    def run(path, greedy, k):
        params = SamplingParams(n=1 if greedy else k, temperature=0.0 if greedy else args.temperature,
                                top_p=1.0 if greedy else args.top_p, max_tokens=args.max_new, seed=args.seed,
                                stop=["<|im_end|>", "<|endoftext|>"])
        start = time.time()
        outs = llm.generate(prompts, params)
        ntok = 0
        with open(path, "w") as sink:
            for t, o in zip(tasks, outs):
                for s, c in enumerate(o.outputs):
                    ntok += len(c.token_ids)
                    sink.write(json.dumps({"id": t["id"], "sample": s, "completion": c.text, "n_tokens": len(c.token_ids),
                                           "finished": c.finish_reason == "stop"}) + "\n")
        el = time.time() - start
        print(f"GEN_DONE {path} {len(tasks)}x{1 if greedy else k} {el:.0f}s {ntok/el:.0f} tok/s", flush=True)


    if args.greedy:
        run(args.out, True, 1)
    else:
        run(args.out, False, args.k)
        if args.also_greedy:
            run(args.out.replace(".jsonl", "") + ".greedy.jsonl", True, 1)


if __name__ == "__main__":
    main()
