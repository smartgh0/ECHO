#!/usr/bin/env python3
"""Stream, clean, deduplicate, and shard a bounded Echo capability corpus.

This script is designed to run on the training pod. It never downloads an
entire upstream corpus intentionally: each category has a token quota and
streaming stops when that quota is met.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import html
import json
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from echo_tokenizer import EchoTokenizer


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DOC_MARKER = "@@DOC@@"
SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
TAG_RE = re.compile(r"<(?:script|style)[^>]*>.*?</(?:script|style)>|<[^>]+>", re.I | re.S)
SECRET_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|AKIA[0-9A-Z]{16}"
    r"|(?:api[_-]?key|secret[_-]?key|password)\s*[:=]\s*['\"][^'\"]{12,}",
    re.I,
)
CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.I | re.S)


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    category: str
    license: str
    configs: tuple[str | None, ...] = (None,)
    mode: str = "raw"  # raw, sft, summary
    terms: str = ""
    revision: str = "main"


SOURCES = (
    Source(
        "fineweb_edu",
        "HuggingFaceFW/fineweb-edu",
        "general",
        "ODC-By-1.0",
        ("sample-10BT", "sample-100BT", None),
        terms="Common Crawl Terms of Use also apply.",
    ),
    Source(
        "fineweb_edu_fallback",
        "HuggingFaceFW/fineweb-edu",
        "general",
        "ODC-By-1.0",
        ("sample-10BT",),
        terms="Fallback after the attempted Dolma allocation; Common Crawl terms apply.",
    ),
    Source(
        "dolma",
        "allenai/dolma",
        "general",
        "ODC-By-1.0",
        ("v1_7", "v1_6-sample", None),
        terms="Underlying source terms also apply.",
    ),
    Source(
        "opencoder_fineweb_code",
        "OpenCoder-LLM/opc-fineweb-code-corpus",
        "code",
        "MIT",
        (None,),
    ),
    Source(
        "opencoder_annealing",
        "OpenCoder-LLM/opc-annealing-corpus",
        "code",
        "MIT",
        ("synthetic_code_snippet", "synthetic_qa", "algorithmic_corpus"),
    ),
    Source(
        "opencoder_sft2",
        "OpenCoder-LLM/opc-sft-stage2",
        "code",
        "MIT",
        ("educational_instruct", "evol_instruct", "mceval_instruct", "package_instruct"),
        mode="sft",
    ),
    Source(
        "smollm_python_edu",
        "HuggingFaceTB/smollm-corpus",
        "code",
        "ODC-By-1.0 / per-file upstream licenses",
        ("python-edu",),
    ),
    Source(
        "finemath",
        "HuggingFaceTB/finemath",
        "math",
        "ODC-By-1.0",
        ("finemath-4plus", "finemath-3plus", None),
        terms="Common Crawl Terms of Use also apply.",
    ),
    Source(
        "megamath",
        "LLM360/MegaMath",
        "math",
        "ODC-By-1.0",
        ("megamath-web-pro", "megamath-web", None),
    ),
    Source(
        "pes2o",
        "allenai/peS2o",
        "science",
        "ODC-By-1.0",
        ("v2", None),
        terms="Open-access paper source terms remain applicable.",
    ),
    Source(
        "open_thoughts3",
        "open-thoughts/OpenThoughts3-1.2M",
        "reasoning",
        "Apache-2.0",
        (None,),
        mode="sft",
    ),
    Source(
        "govreport",
        "launch/gov_report",
        "summary",
        "Not specified by the dataset card",
        ("plain_text", None),
        mode="summary",
        terms="Research-only inclusion; original dataset card does not state a license.",
    ),
    Source(
        "billsum",
        "FiscalNote/billsum",
        "summary",
        "CC0-1.0",
        (None,),
        mode="summary",
    ),
    Source(
        "fineweb_grounded_summary",
        "HuggingFaceFW/fineweb-edu",
        "summary",
        "ODC-By-1.0",
        ("sample-10BT",),
        mode="lead_summary",
        terms="Derived body-to-lead examples; Common Crawl Terms of Use also apply.",
    ),
    Source(
        "xlam_function_calling",
        "Salesforce/xlam-function-calling-60k",
        "tools",
        "CC-BY-4.0",
        ("dataset", None),
        mode="sft",
        terms="Included only when existing Hugging Face authentication permits access.",
    ),
)


CATEGORY_FRACTIONS = {
    "general": 0.27,
    "code": 0.23,
    "math": 0.15,
    "science": 0.15,
    "reasoning": 0.10,
    "summary": 0.08,
    "tools": 0.019,
    "identity": 0.001,
}


HELD_OUT_FRAGMENTS = (
    "billing_nonce",
    "ticket_lock",
    "webhook_secret",
    "launch_window.txt",
    "flags.ini",
    "rotate_right(nums",
    "longest_common_prefix(strs",
    "roman_to_int(s)",
    "chunk_list(items",
    "hamming(a, b)",
    "5(y+2)-3=2y+11",
    "dedupe_preserve_order",
    "ops_memo.txt",
)
ACTIVE_HELD_OUT_FRAGMENTS = list(HELD_OUT_FRAGMENTS)


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(
    value: Any,
    *,
    min_chars: int = 160,
    max_chars: int = 120_000,
) -> str | None:
    if not isinstance(value, str):
        return None
    text = html.unescape(value).replace("\x00", " ")
    text = TAG_RE.sub(" ", text)
    text = "\n".join(SPACE_RE.sub(" ", line).strip() for line in text.splitlines())
    text = BLANK_RE.sub("\n\n", text).strip()
    if len(text) < min_chars:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
        sentence_end = max(text.rfind(". "), text.rfind("\n"))
        if sentence_end > max_chars // 2:
            text = text[: sentence_end + 1]
    if SECRET_RE.search(text):
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    ascii_like = sum(ord(ch) < 128 for ch in text)
    if printable / max(len(text), 1) < 0.98 or ascii_like / max(len(text), 1) < 0.72:
        return None
    lowered = text.lower()
    if any(fragment.lower() in lowered for fragment in ACTIVE_HELD_OUT_FRAGMENTS):
        return None
    return text


def first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def nested_text(value: Any) -> str | None:
    parts: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for key in ("section_title", "title", "paragraphs", "text", "subsections"):
                if key in item:
                    visit(item[key])

    visit(value)
    return "\n\n".join(parts) if parts else None


def messages_to_turn(record: dict[str, Any]) -> str | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        messages = record.get("conversations")
    if isinstance(messages, list):
        user = None
        assistant = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", message.get("from", ""))).lower()
            content = message.get("content", message.get("value"))
            if role in {"user", "human"} and isinstance(content, str) and user is None:
                user = content
            elif role in {"assistant", "model", "gpt"} and isinstance(content, str):
                assistant = content
                if user:
                    break
        if user and assistant:
            return f"user: {user.strip()}\necho: {assistant.strip()}"

    instruction = first_string(
        record,
        ("instruction", "prompt", "question", "problem", "query", "input"),
    )
    output = first_string(
        record,
        ("output", "response", "answer", "solution", "completion", "generation"),
    )
    if instruction and output:
        return f"user: {instruction.strip()}\necho: {output.strip()}"
    return None


def tool_record_to_turn(record: dict[str, Any]) -> str | None:
    query = record.get("query")
    tools = record.get("tools")
    answers = record.get("answers")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except json.JSONDecodeError:
            return None
    if not isinstance(query, str) or not isinstance(tools, list) or not isinstance(answers, list):
        return None
    valid_names = {
        str(tool.get("name"))
        for tool in tools
        if isinstance(tool, dict) and tool.get("name")
    }
    calls = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        name = str(answer.get("name", ""))
        arguments = answer.get("arguments")
        if name not in valid_names or not isinstance(arguments, dict):
            continue
        rendered = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            for key, value in sorted(arguments.items())
        )
        calls.append(f"tool: {name}" + (f" {rendered}" if rendered else ""))
    if not calls:
        return None
    schema = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return (
        "user: Available tools: "
        + schema
        + "\nRequest: "
        + query.strip()
        + "\necho: "
        + "\n".join(calls)
    )


def summary_to_turn(record: dict[str, Any]) -> str | None:
    document = first_string(
        record,
        ("report", "document", "text", "article", "bill", "input", "body"),
    )
    summary = first_string(
        record,
        ("summary", "highlights", "abstract", "target"),
    )
    if not document and "reports" in record:
        document = nested_text(record["reports"])
    if not summary and "summary" in record:
        summary = nested_text(record["summary"])
    if not document or not summary:
        return None
    # Echo has a 2048-token context. Keep source+target inside a useful envelope.
    document = document[:6_000]
    summary = summary[:1_500]
    return (
        "user: Analyze the following source and summarize its key claims accurately. "
        "Do not add facts that are absent from the source.\n\n"
        f"{document.strip()}\n"
        f"echo: {summary.strip()}"
    )


def raw_to_lead_summary(record: dict[str, Any]) -> str | None:
    document = raw_from_record(record)
    if not document or len(document) < 900:
        return None
    boundary = re.search(r"(?<=[.!?])\s+", document)
    cut = boundary.start() if boundary and boundary.start() >= 120 else min(600, len(document) // 3)
    lead = document[:cut].strip()
    body = document[cut:].strip()[:6_000]
    if len(lead) < 80 or len(body) < 400:
        return None
    return (
        "user: Summarize the following source faithfully without adding outside facts.\n\n"
        f"{body}\n"
        f"echo: {lead[:1_200]}"
    )


def raw_from_record(record: dict[str, Any]) -> str | None:
    value = first_string(
        record,
        (
            "text",
            "content",
            "document",
            "body",
            "paper",
            "article",
            "code",
            "source",
        ),
    )
    if value:
        return value
    # Some corpora place useful text in a list of paragraphs.
    for key in ("paragraphs", "sections"):
        value = record.get(key)
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    piece = first_string(item, ("text", "content", "body"))
                    if piece:
                        parts.append(piece)
            if parts:
                return "\n\n".join(parts)
    return None


def record_to_text(record: dict[str, Any], mode: str, category: str) -> str | None:
    if category == "tools":
        converted = tool_record_to_turn(record)
        if converted:
            return converted
    if mode == "sft":
        return messages_to_turn(record)
    if mode == "summary":
        return summary_to_turn(record)
    if mode == "lead_summary":
        return raw_to_lead_summary(record)
    return raw_from_record(record)


def python_is_parseable(text: str) -> bool:
    if "user:" not in text or "echo:" not in text:
        return True
    answer = text.split("echo:", 1)[1]
    fenced = CODE_FENCE_RE.search(answer)
    code = fenced.group(1).strip() if fenced else answer.strip()
    if not any(token in code for token in ("def ", "class ", "import ", "from ")):
        return True
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


class SeenHashes:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen (digest TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")

    def add(self, digest: str) -> bool:
        try:
            self.connection.execute("INSERT INTO seen(digest) VALUES (?)", (digest,))
            return True
        except sqlite3.IntegrityError:
            return False

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class ShardWriter:
    def __init__(self, root: Path, category: str, max_bytes: int):
        self.root = root
        self.category = category
        self.max_bytes = max_bytes
        self.index = 0
        self.handle = None
        self.bytes_in_shard = 0
        self.paths: list[str] = []

    def _open(self) -> None:
        if self.handle:
            self.handle.close()
        path = self.root / f"cap_{self.category}_{self.index:04d}.txt"
        self.index += 1
        self.handle = path.open("w", encoding="utf-8")
        self.paths.append(str(path))
        self.bytes_in_shard = 0

    def write(self, text: str) -> int:
        encoded_size = len(text.encode("utf-8")) + len(DOC_MARKER) + 4
        if self.handle is None or self.bytes_in_shard + encoded_size > self.max_bytes:
            self._open()
        assert self.handle is not None
        self.handle.write(f"{DOC_MARKER}\n{text.rstrip()}\n\n")
        self.bytes_in_shard += encoded_size
        return encoded_size

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


def load_stream(source: Source) -> tuple[Iterator[dict[str, Any]], str | None, str]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    if source.repo == "allenai/dolma":
        import requests

        index_url = (
            f"https://huggingface.co/datasets/{source.repo}/resolve/"
            f"{source.revision}/urls/v1_7.txt"
        )
        response = requests.get(index_url, timeout=90)
        response.raise_for_status()
        urls = [
            line.strip()
            for line in response.text.splitlines()
            if any(
                marker in line
                for marker in (
                    "/books/",
                    "/wiki/",
                    "/wikiref_megawika/",
                )
            )
        ]
        # The category quota stops iteration; a bounded URL list also prevents
        # accidental enumeration of the full multi-terabyte release.
        dataset = load_dataset(
            "json",
            data_files={"train": urls[:96]},
            split="train",
            streaming=True,
        )
        revision = HfApi().dataset_info(source.repo, revision=source.revision).sha
        return iter(dataset), "v1_7-books-wiki-reference-direct", revision

    if source.repo == "launch/gov_report":
        files = (
            f"hf://datasets/{source.repo}@{source.revision}/data/crs_train.jsonl",
            f"hf://datasets/{source.repo}@{source.revision}/data/gao_train.jsonl",
        )
        dataset = load_dataset(
            "json",
            data_files={"train": list(files)},
            split="train",
            streaming=True,
        )
        revision = HfApi().dataset_info(source.repo, revision=source.revision).sha
        return iter(dataset), "plain_text-direct-json", revision

    if source.repo == "allenai/peS2o":
        data_files = (
            f"hf://datasets/{source.repo}@{source.revision}/"
            "data/v2/train-*.json.gz"
        )
        dataset = load_dataset(
            "json",
            data_files={"train": data_files},
            split="train",
            streaming=True,
        )
        revision = HfApi().dataset_info(source.repo, revision=source.revision).sha
        return iter(dataset), "v2-direct-json", revision

    if source.repo == "LLM360/MegaMath":
        for folder in source.configs:
            if folder is None:
                continue
            try:
                data_files = (
                    f"hf://datasets/{source.repo}@{source.revision}/"
                    f"{folder}/*.parquet"
                )
                dataset = load_dataset(
                    "parquet",
                    data_files={"train": data_files},
                    split="train",
                    streaming=True,
                )
                revision = HfApi().dataset_info(
                    source.repo, revision=source.revision
                ).sha
                return iter(dataset), folder, revision
            except Exception:
                continue

    errors = []
    for config in source.configs:
        try:
            kwargs = {
                "path": source.repo,
                "split": "train",
                "streaming": True,
                "trust_remote_code": False,
                "revision": source.revision,
            }
            if config is not None:
                kwargs["name"] = config
            dataset = load_dataset(**kwargs)
            try:
                resolved_revision = HfApi().dataset_info(
                    source.repo, revision=source.revision
                ).sha
            except Exception:
                resolved_revision = source.revision
            return iter(dataset), config, resolved_revision
        except Exception as exc:  # upstream schemas/access can change
            errors.append(f"{config!r}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def token_counts(tokenizer: EchoTokenizer, texts: list[str], threads: int) -> list[int]:
    processor = tokenizer.processor
    try:
        encoded = processor.encode(texts, out_type=int, num_threads=threads)
        return [len(ids) for ids in encoded]
    except (TypeError, RuntimeError):
        return [len(processor.encode(text, out_type=int)) for text in texts]


def synthetic_tool_turns(count: int, seed: int) -> Iterator[str]:
    """Generate schema-exact examples with slots disjoint from held-out eval."""
    rng = random.Random(seed)
    verbs = ("delete", "erase", "remove", "purge", "drop", "invalidate")
    nouns = ("cache", "entry", "key", "cached value")
    prefixes = ("acct", "build", "deploy", "queue", "feature", "auth", "report")
    suffixes = ("token", "nonce", "lock", "state", "flag", "draft", "index")
    file_prefixes = ("release", "service", "worker", "audit", "deploy", "runtime")
    extensions = ("txt", "cfg", "ini", "json", "log", "md")
    values = (
        "enabled=1",
        "debug=false",
        "window=04:30 UTC",
        '{"status":"ready"}',
        "retry_count=3",
        "owner=platform team",
    )
    for i in range(count):
        kind = i % 3
        if kind == 0:
            key = f"{rng.choice(prefixes)}_{rng.choice(suffixes)}_{i:06x}"
            user = (
                f"{rng.choice(verbs).capitalize()} the {rng.choice(nouns)} "
                f"`{key}` permanently using the correct tool."
            )
            yield f"user: {user}\necho: tool: cache_delete key={key}"
        elif kind == 1:
            name = f"{rng.choice(file_prefixes)}_{i:06x}.{rng.choice(extensions)}"
            value = rng.choice(values)
            user = f"Create `{name}` with the exact content: {value}"
            yield f"user: {user}\necho: tool: file_write filename={name} content={value}"
        else:
            command = rng.choice(("ls -la", "ls -1", "pwd", "find . -maxdepth 1 -type f"))
            user = rng.choice(
                (
                    f"Run `{command}` with the shell tool and stop after the tool call.",
                    f"Use a shell tool to execute {command}. Do not invent its output.",
                    f"Execute this in the current directory using a tool: {command}",
                )
            )
            yield f'user: {user}\necho: tool: shell_run command="{command}"'


def synthetic_identity_turns(count: int) -> Iterator[str]:
    prompts = (
        "Who are you and who trained you?",
        "State your name, primary strengths, and trainer.",
        "Introduce yourself briefly.",
        "What model are you?",
    )
    answers = (
        "I'm Echo, a domain-focused AI model for coding, agentic tool calling, "
        "analysis, and general knowledge. I was trained by Solomon Nyamekye.",
        "My name is Echo. I specialize in coding and tool use while retaining broad "
        "knowledge and analytical skills, and I was trained by Solomon Nyamekye.",
    )
    for i in range(count):
        yield f"user: {prompts[i % len(prompts)]}\necho: {answers[i % len(answers)]}"


def iter_local_documents(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    parts: list[str] = []
    chars = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                parts.append(line.rstrip())
                chars += len(line)
            if (not line.strip() and chars >= 160) or chars >= 120_000:
                yield "\n".join(parts)
                parts.clear()
                chars = 0
    if parts:
        yield "\n".join(parts)


def iter_local_summaries(path: Path) -> Iterator[str]:
    for document in iter_local_documents(path):
        if len(document) < 900:
            continue
        first_break = document.find("\n")
        first_sentence = re.search(r"(?<=[.!?])\s+", document)
        candidates = [index for index in (first_break, first_sentence.start() if first_sentence else -1) if index > 120]
        cut = min(candidates) if candidates else min(600, len(document) // 3)
        lead = document[:cut].strip()
        body = document[cut:].strip()[:6_000]
        if len(lead) < 80 or len(body) < 400:
            continue
        yield (
            "user: Summarize the following source faithfully without adding outside facts.\n\n"
            f"{body}\n"
            f"echo: {lead[:1_200]}"
        )


def extend_decontamination_from_evals(root: Path) -> None:
    for path in root.glob("echo_eval*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = re.sub(r"\s+", " ", node.value).strip()
            if 30 <= len(value) <= 600 and value.count(" ") >= 5:
                ACTIVE_HELD_OUT_FRAGMENTS.append(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--shard-mb", type=int, default=128)
    parser.add_argument("--batch-docs", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--allcombined", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extend_decontamination_from_evals(Path(__file__).resolve().parent)
    tokenizer = EchoTokenizer(args.tokenizer)
    seen = SeenHashes(output / "dedupe.sqlite3")
    quotas = {
        category: int(args.target_tokens * fraction)
        for category, fraction in CATEGORY_FRACTIONS.items()
    }
    accepted_tokens = {category: 0 for category in quotas}
    accepted_docs = {category: 0 for category in quotas}
    rejected = {
        category: {"quality": 0, "duplicate": 0, "code_parse": 0}
        for category in quotas
    }
    writers = {
        category: ShardWriter(output, category, args.shard_mb * 1024 * 1024)
        for category in quotas
    }
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_tokens": args.target_tokens,
        "tokenizer": args.tokenizer,
        "category_fractions": CATEGORY_FRACTIONS,
        "sources": [],
        "decontamination_fragment_count": len(ACTIVE_HELD_OUT_FRAGMENTS),
        "decontamination_sha256": hashlib.sha256(
            "\n".join(sorted(ACTIVE_HELD_OUT_FRAGMENTS)).encode("utf-8")
        ).hexdigest(),
    }

    def ingest(
        category: str,
        records: Iterable[str],
        source_name: str,
        source_token_limit: int | None = None,
    ) -> dict[str, int]:
        stats = {"seen": 0, "accepted": 0, "tokens": 0, "bytes": 0}
        batch: list[str] = []
        source_target = quotas[category]
        if source_token_limit is not None:
            source_target = min(
                source_target,
                accepted_tokens[category] + source_token_limit,
            )

        def consume(items: list[str]) -> bool:
            if not items:
                return False
            counts = token_counts(tokenizer, items, args.tokenizer_threads)
            for text, count in zip(items, counts):
                if accepted_tokens[category] >= source_target:
                    return True
                remaining = source_target - accepted_tokens[category]
                if count > remaining and accepted_tokens[category] > source_target * 0.98:
                    return True
                stats["bytes"] += writers[category].write(text)
                stats["accepted"] += 1
                stats["tokens"] += count
                accepted_docs[category] += 1
                accepted_tokens[category] += count
            seen.commit()
            return accepted_tokens[category] >= source_target

        for raw in records:
            stats["seen"] += 1
            minimum = 40 if category in {"tools", "identity"} else 160
            text = clean_text(raw, min_chars=minimum)
            if text is None:
                rejected[category]["quality"] += 1
                continue
            if category == "code" and not python_is_parseable(text):
                rejected[category]["code_parse"] += 1
                continue
            digest = normalized_hash(text)
            if not seen.add(digest):
                rejected[category]["duplicate"] += 1
                continue
            batch.append(text)
            if len(batch) >= args.batch_docs:
                if consume(batch):
                    batch.clear()
                    break
                batch.clear()
        consume(batch)
        print(
            f"[{category}] {source_name}: accepted={stats['accepted']:,} "
            f"tokens={stats['tokens']:,}; total={accepted_tokens[category]:,}/"
            f"{quotas[category]:,}",
            flush=True,
        )
        stats["rejected"] = stats["seen"] - stats["accepted"]
        return stats

    try:
        # Local encyclopedic corpus first.
        if args.allcombined:
            path = Path(args.allcombined)
            stats = ingest(
                "general",
                iter_local_documents(path),
                "AllCombined",
                int(quotas["general"] * 0.25),
            )
            manifest["sources"].append(
                {
                    "name": "AllCombined",
                    "path": str(path),
                    "category": "general",
                    "license": "Source-specific; local corpus",
                    "status": "loaded" if path.exists() else "missing",
                    **stats,
                }
            )
            stats = ingest(
                "summary",
                iter_local_summaries(path),
                "AllCombined_body_to_lead",
                int(quotas["summary"] * 0.35),
            )
            manifest["sources"].append(
                {
                    "name": "AllCombined_body_to_lead",
                    "path": str(path),
                    "category": "summary",
                    "license": "Source-specific; local corpus",
                    "terms": "Derived body-to-lead examples; same terms as AllCombined.",
                    "status": "loaded" if path.exists() else "missing",
                    **stats,
                }
            )

        # Synthetic schema-exact anchors are generated before generic tool data.
        stats = ingest(
            "tools",
            synthetic_tool_turns(900_000, args.seed),
            "echo_schema_synthetic",
            int(quotas["tools"] * 0.55),
        )
        manifest["sources"].append(
            {
                "name": "echo_schema_synthetic",
                "category": "tools",
                "license": "Project-generated",
                "status": "loaded",
                **stats,
            }
        )
        stats = ingest("identity", synthetic_identity_turns(30_000), "echo_identity")
        manifest["sources"].append(
            {
                "name": "echo_identity",
                "category": "identity",
                "license": "Project-generated",
                "status": "loaded",
                **stats,
            }
        )

        source_priority = {
            "opencoder_sft2": 0,
            "opencoder_annealing": 1,
            "opencoder_fineweb_code": 2,
            "smollm_python_edu": 3,
            "fineweb_edu": 0,
            "dolma": 1,
            "fineweb_edu_fallback": 2,
            "govreport": 0,
            "billsum": 1,
            "fineweb_grounded_summary": 2,
        }
        source_caps = {
            "opencoder_sft2": 0.25,
            "opencoder_annealing": 0.25,
            "fineweb_edu": 0.60,
            "dolma": 0.20,
            "finemath": 0.70,
            "govreport": 0.25,
            "billsum": 0.15,
        }
        ordered_sources = sorted(
            SOURCES,
            key=lambda source: (
                list(quotas).index(source.category),
                source_priority.get(source.name, 10),
            ),
        )
        for source in ordered_sources:
            if accepted_tokens[source.category] >= quotas[source.category]:
                manifest["sources"].append(
                    {**asdict(source), "status": "quota_already_met"}
                )
                continue
            entry: dict[str, Any] = asdict(source)
            stream = None
            try:
                stream, selected_config, resolved_revision = load_stream(source)

                def converted() -> Iterator[str]:
                    empty_streak = 0
                    for record in stream:
                        if not isinstance(record, dict):
                            continue
                        text = record_to_text(record, source.mode, source.category)
                        if text:
                            empty_streak = 0
                            # Never train the model to continue a fake tool result.
                            if source.category == "tools":
                                text = text.split("\ntool_result", 1)[0].rstrip()
                            yield text
                        else:
                            empty_streak += 1
                            if empty_streak >= 1_000:
                                print(
                                    f"STOP {source.name}: 1,000 consecutive rows "
                                    "had no usable text fields",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                break

                cap = source_caps.get(source.name)
                stats = ingest(
                    source.category,
                    converted(),
                    source.name,
                    int(quotas[source.category] * cap) if cap is not None else None,
                )
                entry.update(
                    {
                        "status": "loaded",
                        "selected_config": selected_config,
                        "resolved_revision": resolved_revision,
                        **stats,
                    }
                )
            except Exception as exc:
                entry.update(
                    {
                        "status": "skipped",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"SKIP {source.name}: {entry['error']}", file=sys.stderr, flush=True)
            finally:
                if stream is not None and hasattr(stream, "close"):
                    try:
                        stream.close()
                    except Exception:
                        pass
                stream = None
                gc.collect()
            manifest["sources"].append(entry)
    finally:
        for writer in writers.values():
            writer.close()
        seen.close()

    manifest["accepted_tokens"] = accepted_tokens
    manifest["accepted_docs"] = accepted_docs
    manifest["quotas"] = quotas
    manifest["rejected"] = rejected
    manifest["total_accepted_tokens"] = sum(accepted_tokens.values())
    manifest["shards"] = {
        category: writer.paths for category, writer in writers.items()
    }
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted_tokens": accepted_tokens}, indent=2), flush=True)
    print(f"TOTAL {sum(accepted_tokens.values()):,} tokens -> {output}", flush=True)


if __name__ == "__main__":
    main()
