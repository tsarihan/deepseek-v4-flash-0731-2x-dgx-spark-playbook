#!/usr/bin/env python3
"""Long-context needle-retrieval probe for DeepSeek-V4-Flash-0731.

Builds a prompt of ~N tokens with a needle buried at a given depth, verifies the
real token count via the server's /tokenize endpoint, then checks the model
actually retrieves it. Reports TTFT and prefill throughput.
"""
import argparse, json, time, urllib.request

FILLER = ("The maintenance log records routine calibration of the sensor array. "
          "Ambient conditions remained nominal throughout the observation window. ")


def post(url, body, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ntokens(base, model, text):
    return post(base.removesuffix("/v1") + "/tokenize",
                {"model": model, "prompt": text})["count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--depth", type=float, default=0.5, help="0..1 needle depth")
    ap.add_argument("--secret", default="ARGON-SEVEN-FOUR-ZERO")
    a = ap.parse_args()

    needle = (f"\n\nIMPORTANT FACT: the vault authorization code is {a.secret}. "
              f"Remember it.\n\n")

    # grow filler to hit the target token count
    approx = max(1, a.tokens // 22)
    body = FILLER * approx
    count = ntokens(a.base_url, a.model, body)
    while count < a.tokens * 0.98:
        body += FILLER * max(1, (a.tokens - count) // 22)
        count = ntokens(a.base_url, a.model, body)
    cut = int(len(body) * a.depth)
    prompt = (body[:cut] + needle + body[cut:] +
              "\n\nQuestion: What is the vault authorization code stated above? "
              "Answer with the code only.")
    total = ntokens(a.base_url, a.model, prompt)
    print(f"prompt tokens: {total:,} (target {a.tokens:,}, needle at depth {a.depth})",
          flush=True)

    req_body = {"model": a.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True, "stream_options": {"include_usage": True},
                "temperature": 0.0, "max_tokens": 256}
    req = urllib.request.Request(f"{a.base_url}/chat/completions",
                                 data=json.dumps(req_body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; out = []; usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            ch = ev.get("choices") or []
            d = ch[0].get("delta", {}) if ch else {}
            piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
            if first is None and piece:
                first = time.perf_counter()
            out.append(d.get("content") or "")
            if ev.get("usage"):
                usage = ev["usage"]
    done = time.perf_counter()
    text = "".join(out)
    ttft = (first or done) - t0
    ok = a.secret.lower().replace("-", "") in text.lower().replace("-", "")
    print(f"TTFT: {ttft:.1f}s   prefill: {total/ttft:,.0f} tok/s   "
          f"total: {done-t0:.1f}s")
    print(f"usage: {usage}")
    print(f"answer: {text.strip()[:200]!r}")
    print(f"NEEDLE RETRIEVED: {ok}")


if __name__ == "__main__":
    main()
