#!/usr/bin/env python3
"""Distill agent/tool-calling data from GLM-5.2:cloud for Echo.

Uses a system prompt that forces the teacher to respond in tool-call format:
think: <reasoning>
tool: <tool_name> <args>
tool_result: <simulated result>
echo: <response to user>
"""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request

from generate_agent_prompts import SYSTEM_PROMPT

DEFAULT_MODEL = "glm-5.2:cloud"
DEFAULT_API = "http://127.0.0.1:11434"
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pipeline", "input", "distill_only", "agent_data.txt",
)


def query_ollama(prompt, model, api, max_tokens=500, timeout=300, retries=5):
    """Query with the agent system prompt and retry on empty responses."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": max_tokens, "top_p": 0.9, "top_k": 40},
    }
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"{api}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()
            if content and len(content) > 20:
                return content
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise
    return None


def load_prompts(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--prompts", default="agent_prompts.txt")
    parser.add_argument("--count", type=int, default=275)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if prompts is None:
        print(f"No prompts file found at {args.prompts}")
        return
    prompts = prompts[:args.count]

    print(f"Teacher model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    success = 0
    total_chars = 0
    start_time = time.time()

    with open(args.output, "w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for prompt in prompts:
                future = pool.submit(query_ollama, prompt, args.model, args.api,
                                     args.max_tokens, args.timeout, args.retries)
                futures.append((future, prompt))

            for future, prompt in futures:
                try:
                    response = future.result(timeout=600)
                except Exception:
                    continue
                if response:
                    entry = f"user: {prompt}\necho: {response}\n"
                    handle.write(entry)
                    handle.flush()
                    total_chars += len(entry)
                    success += 1
                    if success % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = success / max(elapsed, 1)
                        print(f"  [{success}/{len(prompts)}] {total_chars:,} chars | "
                              f"{rate:.1f}/s", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDistilled {success}/{len(prompts)} agent Q&A pairs")
    print(f"Wrote {total_chars:,} chars ({total_chars/1024:.0f} KB) to {args.output}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nUpload {args.output} to RunPod before training:")
    print(f"  scp -P <port> {args.output} root@<runpod-ip>:/workspace/ECHO/pipeline/input/distill_only/")


if __name__ == "__main__":
    main()
