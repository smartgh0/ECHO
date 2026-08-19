#!/usr/bin/env python3
"""Distill conversational data from a local Ollama teacher model into Echo.

Supports parallel API calls for large-scale distillation.

Example:
    python3 distill_echo.py --model glm-5.2:cloud --count 200000 --workers 10
    python3 distill_echo.py --prompts prompts.txt --count 5000 --workers 5
"""

import argparse
import concurrent.futures
import json
import os
import random
import time
import urllib.request

DEFAULT_MODEL = "glm-5.2:cloud"
DEFAULT_API = "http://127.0.0.1:11434"
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pipeline", "input", "distill_only", "distilled_data.txt",
)

PROMPT_TEMPLATES = [
    "Explain {topic} in simple terms.",
    "What is {topic}?",
    "How does {topic} work?",
    "Why is {topic} important?",
    "Describe {topic}.",
    "Give an example of {topic}.",
    "What are the main concepts in {topic}?",
    "How would you teach {topic} to a beginner?",
    "What is the difference between {topic} and {alternative}?",
    "What are common misconceptions about {topic}?",
    "How has {topic} evolved over time?",
    "What are the practical applications of {topic}?",
    "What are the challenges in {topic}?",
    "How do you get started with {topic}?",
    "What are the key principles of {topic}?",
    "Explain {topic} like you are talking to a child.",
    "What is the future of {topic}?",
    "How does {topic} relate to everyday life?",
    "What skills do you need for {topic}?",
    "What are the benefits of learning {topic}?",
]

TOPICS = [
    "gravity", "photosynthesis", "evolution", "DNA", "quantum physics", "relativity",
    "thermodynamics", "electromagnetism", "the solar system", "black holes", "atoms",
    "molecules", "chemical reactions", "the periodic table", "nuclear energy",
    "climate change", "plate tectonics", "the water cycle", "weather patterns",
    "ocean currents", "ecosystems", "biodiversity", "the immune system", "viruses",
    "bacteria", "cells", "proteins", "enzymes", "neurons", "the brain",
    "genetics", "natural selection", "mutation", "gene expression", "stem cells",
    "neural networks", "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "computer vision", "reinforcement learning",
    "transformers", "attention mechanisms", "embeddings", "gradient descent",
    "backpropagation", "overfitting", "regularization", "dropout", "batch normalization",
    "the internet", "TCP/IP", "DNS", "HTTP", "HTTPS", "REST APIs", "GraphQL",
    "databases", "SQL", "NoSQL", "indexing", "transactions", "ACID properties",
    "cloud computing", "virtualization", "containers", "Docker", "Kubernetes",
    "microservices", "load balancing", "caching", "CDNs", "message queues",
    "encryption", "public key cryptography", "hash functions", "digital signatures",
    "blockchain", "cryptocurrency", "smart contracts", "zero-knowledge proofs",
    "operating systems", "processes", "threads", "memory management", "file systems",
    "compilers", "interpreters", "garbage collection", "type systems", "polymorphism",
    "Python", "JavaScript", "Java", "C++", "Rust", "Go", "TypeScript",
    "object-oriented programming", "functional programming", "recursion",
    "data structures", "algorithms", "sorting", "searching", "graph algorithms",
    "dynamic programming", "greedy algorithms", "divide and conquer",
    "design patterns", "unit testing", "test-driven development", "code review",
    "version control", "Git", "continuous integration", "DevOps", "agile development",
    "debugging", "profiling", "optimization", "concurrency", "parallel programming",
    "async programming", "event-driven programming", "reactive programming",
    "calculus", "linear algebra", "probability", "statistics", "discrete math",
    "set theory", "graph theory", "combinatorics", "number theory", "topology",
    "differential equations", "optimization", "linear programming", "game theory",
    "the Fourier transform", "matrix multiplication", "eigenvalues", "vector spaces",
    "supply and demand", "inflation", "interest rates", "stock markets",
    "venture capital", "startup funding", "business models", "marketing strategy",
    "product management", "agile methodologies", "project management", "risk management",
    "financial analysis", "budgeting", "investing", "compound interest",
    "philosophy", "ethics", "logic", "critical thinking", "the scientific method",
    "history of science", "the Renaissance", "the Industrial Revolution",
    "world wars", "democracy", "capitalism", "socialism", "globalization",
    "psychology", "cognitive science", "behavioral economics", "sociology",
    "music theory", "harmony", "rhythm", "melody", "literature", "poetry",
    "storytelling", "film making", "photography", "painting", "sculpture",
    "nutrition", "exercise", "sleep", "mental health", "meditation",
    "vaccines", "antibiotics", "the placebo effect", "stress management",
    "cooking", "gardening", "writing", "public speaking", "negotiation",
    "time management", "leadership", "teamwork", "communication skills",
    "problem solving", "decision making", "creative thinking",
]

ALTERNATIVES = {
    "neural networks": "traditional programming",
    "machine learning": "deep learning",
    "SQL": "NoSQL",
    "Python": "JavaScript",
    "calculus": "linear algebra",
    "cloud computing": "edge computing",
    "object-oriented programming": "functional programming",
    "REST APIs": "GraphQL",
    "encryption": "hashing",
    "blockchain": "traditional databases",
    "Docker": "virtual machines",
    "microservices": "monoliths",
    "agile development": "waterfall development",
    "unit testing": "integration testing",
    "psychology": "sociology",
    "music theory": "music production",
    "nutrition": "exercise",
    "capitalism": "socialism",
    "philosophy": "science",
    "statistics": "probability",
}


def generate_prompts(count, seed=42):
    rng = random.Random(seed)
    prompts = []
    seen = set()
    attempts = 0
    while len(prompts) < count and attempts < count * 10:
        template = rng.choice(PROMPT_TEMPLATES)
        topic = rng.choice(TOPICS)
        alternative = ALTERNATIVES.get(topic, rng.choice(TOPICS))
        prompt = template.format(topic=topic, alternative=alternative)
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
        attempts += 1
    return prompts


def query_ollama(prompt, model, api, temperature=0.7, max_tokens=300, timeout=300, retries=3):
    """Query the local Ollama model with retry logic."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful, knowledgeable assistant. Answer clearly and concisely. Keep responses under 200 words."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
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
            return data.get("message", {}).get("content", "").strip()
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise


def load_prompts(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--prompts", default=None, help="file with prompts, one per line")
    parser.add_argument("--count", type=int, default=10000, help="number of prompts to distill")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--workers", type=int, default=5, help="parallel API requests")
    parser.add_argument("--timeout", type=int, default=300, help="API timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="retry attempts per prompt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if prompts is None:
        prompts = generate_prompts(args.count, args.seed)
    prompts = prompts[:args.count]

    print(f"Teacher model: {args.model}")
    print(f"Prompts: {len(prompts):,}")
    print(f"Workers: {args.workers} parallel requests")
    print(f"Output: {args.output}")
    print(f"Estimated time: {len(prompts) * 1.5 / args.workers / 3600:.1f} hours")
    print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    success = 0
    total_chars = 0
    start_time = time.time()

    with open(args.output, "w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for index, prompt in enumerate(prompts):
                future = pool.submit(
                    query_ollama, prompt, args.model, args.api,
                    args.temperature, args.max_tokens, args.timeout, args.retries,
                )
                futures.append((future, index, prompt))

            for future, index, prompt in futures:
                try:
                    response = future.result(timeout=300)
                except Exception as exc:
                    continue
                if response:
                    entry = f"user: {prompt}\necho: {response}\n"
                    handle.write(entry)
                    handle.flush()
                    total_chars += len(entry)
                    success += 1
                    if success % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = success / max(elapsed, 1)
                        remaining = (len(prompts) - success) / max(rate, 0.1)
                        print(f"  [{success}/{len(prompts)}] {total_chars:,} chars | "
                              f"{rate:.1f}/s | ETA {remaining/60:.0f}min", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDistilled {success:,} Q&A pairs")
    print(f"Wrote {total_chars:,} chars ({total_chars/1024/1024:.1f} MB) to {args.output}")
    print(f"Time: {elapsed/60:.1f} minutes ({success/max(elapsed,1):.1f} pairs/sec)")
    print(f"\nNext: rebuild tokens and train:")
    print(f"  python3 train_domain.py --input-dir pipeline/input/distill_only \\")
    print(f"    --output-dir domain_brain --profile coherent-150m --rebuild-tokens --steps 20000 --seq-len 256")


if __name__ == "__main__":
    main()