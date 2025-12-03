import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from openai import BadRequestError
import tiktoken

from src.VectorBase import VectorStore
from src.utils import ReadFiles
from src.LLM import OpenAIChat, RAG_PROMPT_TEMPLATE
from src.Embeddings import OpenAIEmbedding
from src.token_estimator import estimate_tokens

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DOCS_PATH", ROOT_DIR / "AI_database")).resolve()
STORAGE_DIR = Path(os.getenv("VECTOR_STORE_PATH", ROOT_DIR / "storage")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "output")).resolve()
TEMPLATE_PATH = Path(os.getenv("QUESTION_TEMPLATE_PATH", BASE_DIR / "示例模板.json")).resolve()
ANSWER_PATH = Path(os.getenv("ANSWER_OUTPUT_PATH", OUTPUT_DIR / "answers_and_contexts.json")).resolve()
PERFORMANCE_PATH = Path(os.getenv("PERFORMANCE_OUTPUT_PATH", OUTPUT_DIR / "performance.json")).resolve()
TOP_K = int(os.getenv("TOP_K", "3"))
DOCS_LIMIT = int(os.getenv("DOCS_LIMIT", "0"))  # 调试用，默认处理全部文档
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "30000"))
MAX_CONTEXTS_PER_QUESTION = int(os.getenv("MAX_CONTEXTS_PER_QUESTION", str(TOP_K)))
_token_encoder = tiktoken.get_encoding("cl100k_base")
_keyword_pattern = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
_source_prefix_pattern = re.compile(r"^【来源：([^】]+)】")
_SOURCE_INDEX: Dict[str, Path] = {}
_SOURCE_TEXT_CACHE: Dict[str, str] = {}

QUESTION_TYPE_CONFIG = {
    "direct": {
        "top_k": 2,
        "max_contexts": 2,
        "token_budget": 1000,
        "expand_sentences": 1,
        "seed_sentences_per_block": 1,
        "per_block_cap": 2
    },
    "lite_analysis": {
        "top_k": 3,
        "max_contexts": 3,
        "token_budget": 3000,
        "expand_sentences": 2,
        "seed_sentences_per_block": 2,
        "per_block_cap": 3
    },
    "deep_analysis": {
        "top_k": 4,
        "max_contexts": 4,
        "token_budget": 5000,
        "expand_sentences": 4,
        "seed_sentences_per_block": 3,
        "per_block_cap": 4
    }
}

_RETRIEVAL_FIELDS = (
    "top_k",
    "max_contexts",
    "max_tokens",
    "expand_sentences",
    "seed_sentences_per_block",
    "per_block_cap"
)


def _ensure_positive(value: int, fallback: int) -> int:
    return fallback if value <= 0 else value


MAX_CONTEXT_TOKENS = _ensure_positive(MAX_CONTEXT_TOKENS, 2000)
MAX_CONTEXTS_PER_QUESTION = _ensure_positive(MAX_CONTEXTS_PER_QUESTION, 1)


def _build_source_index() -> Dict[str, Path]:
    if _SOURCE_INDEX:
        return _SOURCE_INDEX
    for root, _, files in os.walk(DATA_ROOT):
        for file_name in files:
            if file_name.lower().endswith((".txt", ".md", ".pdf")):
                _SOURCE_INDEX[file_name] = Path(root) / file_name
    return _SOURCE_INDEX


def _get_source_text(source_name: str) -> str:
    build_index = _build_source_index()
    source_path = build_index.get(source_name)
    if not source_path:
        return ""
    cache_key = str(source_path)
    if cache_key in _SOURCE_TEXT_CACHE:
        return _SOURCE_TEXT_CACHE[cache_key]
    try:
        text = ReadFiles.read_file_content(cache_key)
    except Exception as exc:  # pragma: no cover - 读文件异常直接记录
        print(f"读取源文件失败：{source_name} -> {exc}")
        text = ""
    _SOURCE_TEXT_CACHE[cache_key] = text
    return text


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_token_encoder.encode(text))


def truncate_by_tokens(text: str, token_limit: int) -> str:
    if token_limit <= 0 or not text:
        return ""
    tokens = _token_encoder.encode(text)
    if len(tokens) <= token_limit:
        return text
    truncated = _token_encoder.decode(tokens[:token_limit]).rstrip()
    return truncated

def classify_question(question: str) -> str:
    if not question:
        return "direct"
    question_lower = question.lower()
    deep_keywords = ["如何", "阐述", "结构", "实现", "比较", "包含", "传递", "目标值"]
    lite_keywords = ["列出", "哪些", "需要", "根据", "说明", "请给出"]
    if any(keyword in question for keyword in deep_keywords):
        return "deep_analysis"
    if any(keyword in question for keyword in lite_keywords):
        return "lite_analysis"
    if "?" in question_lower or "多少" in question or "第几" in question or "何时" in question:
        return "direct"
    return "direct"


def get_question_config(question_type: str) -> Dict[str, int]:
    base = QUESTION_TYPE_CONFIG.get(question_type, QUESTION_TYPE_CONFIG["direct"])
    return {
        "top_k": max(1, min(TOP_K, base["top_k"])),
        "max_contexts": max(1, min(MAX_CONTEXTS_PER_QUESTION, base["max_contexts"])),
        "max_tokens": max(200, min(MAX_CONTEXT_TOKENS, base["token_budget"])),
        "expand_sentences": max(0, base.get("expand_sentences", 0)),
        "seed_sentences_per_block": max(0, base.get("seed_sentences_per_block", 1)),
        "per_block_cap": max(0, base.get("per_block_cap", 0))
    }


def _sanitize_retrieval_value(field: str, value) -> Optional[int]:
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    if field in ("expand_sentences", "seed_sentences_per_block", "per_block_cap"):
        return max(0, numeric_value)
    if field == "max_tokens":
        return max(200, numeric_value)
    return max(1, numeric_value)


def apply_retrieval_overrides(base_config: Dict[str, int], overrides: Optional[Dict[str, int]]) -> Dict[str, int]:
    if not overrides:
        return base_config
    merged = dict(base_config)
    for field in _RETRIEVAL_FIELDS:
        if field not in overrides:
            continue
        sanitized = _sanitize_retrieval_value(field, overrides[field])
        if sanitized is None:
            continue
        merged[field] = sanitized
    merged["top_k"] = max(1, min(TOP_K, merged["top_k"]))
    merged["max_contexts"] = max(1, min(MAX_CONTEXTS_PER_QUESTION, merged["max_contexts"]))
    merged["max_tokens"] = max(200, min(MAX_CONTEXT_TOKENS, merged["max_tokens"]))
    return merged


def _extract_keywords(text: str) -> List[str]:
    if not text:
        return []
    keywords: List[str] = []
    tokens = _keyword_pattern.findall(text)
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        lowered = token.lower()
        keywords.append(lowered)
        if any('\u4e00' <= ch <= '\u9fff' for ch in token) and len(token) > 1:
            keywords.extend([ch for ch in token])
    return keywords


def _score_sentence(sentence: str, keywords: List[str]) -> int:
    if not sentence or not keywords:
        return 0
    sentence_lower = sentence.lower()
    score = 0
    for keyword in keywords:
        if not keyword:
            continue
        count = sentence_lower.count(keyword)
        if count:
            score += count * max(len(keyword), 1)
    return score


def expand_context_with_source(question: str, context_blocks: List[str],
                               max_sentences_per_block: int,
                               max_sentence_tokens: int = 160) -> List[str]:
    if max_sentences_per_block <= 0 or not context_blocks:
        return context_blocks
    keywords = _extract_keywords(question)
    if not keywords:
        return context_blocks
    expanded_blocks = list(context_blocks)
    seen_sources = set()
    for block in context_blocks:
        match = _source_prefix_pattern.match(block or "")
        if not match:
            continue
        source_name = match.group(1)
        if not source_name or source_name in seen_sources:
            continue
        source_text = _get_source_text(source_name)
        if not source_text:
            continue
        sentences = ReadFiles.split_sentences(source_text)
        candidates = []
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            score = _score_sentence(sentence, keywords)
            if score <= 0:
                continue
            candidates.append((score, idx, sentence))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected_sentences = []
        for _, _, sentence in candidates[:max_sentences_per_block]:
            trimmed = truncate_by_tokens(sentence, max_sentence_tokens)
            if trimmed:
                selected_sentences.append(trimmed)
        if selected_sentences:
            expanded_blocks.append(f"【来源扩展：{source_name}】\n" + "\n".join(selected_sentences))
            seen_sources.add(source_name)
    return expanded_blocks


def build_limited_context(question: str,
                          context_blocks: List[str],
                          max_tokens: int,
                          seed_sentences_per_block: int,
                          per_block_cap: int) -> str:
    if not context_blocks:
        return "（未检索到相关文本）"

    keywords = _extract_keywords(question)
    entries = []
    block_sources: Dict[int, str] = {}
    for block_idx, block in enumerate(context_blocks):
        block = (block or "").strip()
        if not block:
            continue
        source_line = ""
        if block.startswith("【来源"):
            first_line, _, remainder = block.partition("\n")
            source_line = first_line.strip()
            block = remainder.strip()
        block_sources[block_idx] = source_line
        sentences = ReadFiles.split_sentences(block) or ([block] if block else [])
        for sentence_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            entries.append({
                "block_idx": block_idx,
                "sentence_idx": sentence_idx,
                "text": sentence,
                "score": _score_sentence(sentence, keywords)
            })

    if not entries:
        fallback_text = truncate_by_tokens(context_blocks[0], max_tokens)
        return fallback_text if fallback_text else "（上下文超出限制，已截断）"

    block_entries: Dict[int, List[Dict[str, int]]] = defaultdict(list)
    for entry in entries:
        block_entries[entry["block_idx"]].append(entry)
    for block_list in block_entries.values():
        block_list.sort(key=lambda e: (-e["score"], e["sentence_idx"]))

    ordered_entries = []
    used = set()
    for block_idx in range(len(context_blocks)):
        block_list = block_entries.get(block_idx, [])
        if not block_list:
            continue
        for entry in block_list[:seed_sentences_per_block]:
            entry_id = (entry["block_idx"], entry["sentence_idx"])
            if entry_id in used:
                continue
            ordered_entries.append(entry)
            used.add(entry_id)

    remaining_entries = []
    for block_idx, block_list in block_entries.items():
        for entry in block_list:
            entry_id = (entry["block_idx"], entry["sentence_idx"])
            if entry_id in used:
                continue
            remaining_entries.append(entry)
    remaining_entries.sort(key=lambda e: (-e["score"], e["block_idx"], e["sentence_idx"]))
    ordered_entries.extend(remaining_entries)

    tokens_used = 0
    snippets: List[str] = []
    emitted_source_blocks = set()
    per_block_added = defaultdict(int)

    for entry in ordered_entries:
        if tokens_used >= max_tokens:
            break
        if per_block_cap and per_block_added[entry["block_idx"]] >= per_block_cap:
            continue
        sentence_tokens = count_tokens(entry["text"])
        if sentence_tokens <= 0:
            continue
        additional_tokens = sentence_tokens
        source_line = block_sources.get(entry["block_idx"], "")
        include_source = False
        if source_line and entry["block_idx"] not in emitted_source_blocks:
            source_tokens = count_tokens(source_line)
            additional_tokens += source_tokens
            include_source = True
        if tokens_used + additional_tokens > max_tokens:
            continue
        if include_source:
            snippets.append(source_line)
            tokens_used += count_tokens(source_line)
            emitted_source_blocks.add(entry["block_idx"])
        snippets.append(entry["text"])
        tokens_used += sentence_tokens
        per_block_added[entry["block_idx"]] += 1

    if not snippets:
        fallback_text = truncate_by_tokens(context_blocks[0], max_tokens)
        return fallback_text if fallback_text else "（上下文超出限制，已截断）"

    return "\n\n".join(snippets)


def load_documents() -> List[str]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"未找到 AI_database 目录：{DATA_ROOT}")
    reader = ReadFiles(str(DATA_ROOT))
    if DOCS_LIMIT > 0:
        reader.file_list = reader.file_list[:DOCS_LIMIT]
    docs = reader.get_content(max_token_len=600, cover_content=150)
    if DOCS_LIMIT > 0:
        docs = docs[:DOCS_LIMIT]
    if not docs:
        raise RuntimeError(f"未在 {DATA_ROOT} 中读取到任何文档，请确认目录下包含可解析的文件。")
    return docs


def build_vector_store(docs: List[str], embedding: OpenAIEmbedding) -> VectorStore:
    vector = VectorStore(docs)
    vector.get_vector(EmbeddingModel=embedding)
    vector.persist(path=str(STORAGE_DIR))
    return vector


def load_or_build_vector_store(embedding: OpenAIEmbedding) -> VectorStore:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    document_file = STORAGE_DIR / "document.json"
    vector_file = STORAGE_DIR / "vectors.json"
    if document_file.exists() and vector_file.exists():
        vector = VectorStore()
        vector.load_vector(str(STORAGE_DIR))
        print(f"已加载缓存向量库：{vector_file} / {document_file}")
        return vector
    print("未发现缓存向量库，开始读取 AI_database 并重新生成向量……")
    docs = load_documents()
    return build_vector_store(docs, embedding)


def answer_questions(vector: VectorStore, embedding: OpenAIEmbedding) -> Dict[str, Path]:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"未找到示例模板：{TEMPLATE_PATH}")
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = json.load(f)
    raw_profiles = template.get("profiles") or {}
    profiles = {
        (name or "").strip().lower(): cfg
        for name, cfg in raw_profiles.items()
        if isinstance(cfg, dict)
    }

    chat_model = OpenAIChat(model='DeepSeek')
    answered_items = []
    performance_records: List[Dict[str, object]] = []

    for item in template.get("items", []):
        question = (item.get("question") or "").strip()
        if not question:
            continue
        provided_type = (item.get("question_type") or "").strip().lower()
        question_type = provided_type if provided_type in QUESTION_TYPE_CONFIG else classify_question(question)
        q_config = get_question_config(question_type)
        profile_name = (item.get("profile") or "").strip().lower()
        profile_cfg = profiles.get(profile_name, {})
        profile_retrieval = profile_cfg.get("retrieval_parameters")
        if not isinstance(profile_retrieval, dict):
            profile_retrieval = None
        q_config = apply_retrieval_overrides(q_config, profile_retrieval)
        overrides = None
        candidate_override = item.get("retrieval_parameters")
        if isinstance(candidate_override, dict):
            overrides = candidate_override
        else:
            candidate_override = item.get("retrieval_config")
            overrides = candidate_override if isinstance(candidate_override, dict) else None
        q_config = apply_retrieval_overrides(q_config, overrides)
        prompt_template = item.get("prompt_template") or profile_cfg.get("prompt_template")
        llm_temperature = None
        for candidate_params in (profile_cfg.get("llm_parameters"), item.get("llm_parameters")):
            if not isinstance(candidate_params, dict):
                continue
            if "temperature" in candidate_params and candidate_params["temperature"] is not None:
                try:
                    llm_temperature = float(candidate_params["temperature"])
                except (TypeError, ValueError):
                    continue
        query_k = q_config["top_k"]
        retrieval_start = time.perf_counter()
        raw_contexts = vector.query(question, EmbeddingModel=embedding, k=query_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        max_context_count = min(q_config["max_contexts"], len(raw_contexts)) if raw_contexts else 0
        retrieved_contexts = raw_contexts[:max_context_count] if raw_contexts else []
        expanded_contexts = expand_context_with_source(
            question,
            retrieved_contexts,
            max_sentences_per_block=q_config["expand_sentences"]
        )

        context_limit = q_config["max_tokens"]
        attempts = 0
        generation_start = None
        generation_end = None
        final_prompt_text = ""
        while True:
            context_text = build_limited_context(
                question,
                expanded_contexts,
                context_limit,
                seed_sentences_per_block=q_config["seed_sentences_per_block"],
                per_block_cap=q_config["per_block_cap"]
            )
            prompt_body = (prompt_template or RAG_PROMPT_TEMPLATE)
            formatted_prompt = prompt_body.format(question=question, context=context_text)
            try:
                if generation_start is None:
                    generation_start = time.perf_counter()
                answer = chat_model.chat(
                    question,
                    [],
                    context_text,
                    prompt_template=prompt_template,
                    temperature=llm_temperature
                )
                generation_end = time.perf_counter()
                final_prompt_text = formatted_prompt
                break
            except BadRequestError as err:
                attempts += 1
                print(f"调用 LLM 失败（{question}）：{err}. 准备缩减上下文重新尝试。")
                if context_limit <= 200 or attempts >= 3:
                    raise
                context_limit = max(200, context_limit // 2)
                print(f"上下文 token 限制调整为 {context_limit}，第 {attempts} 次重试。")
        answered_items.append({
            "question": question,
            "retrieved_contexts": retrieved_contexts,
            "answer": answer
        })
        prompt_tokens = estimate_tokens(final_prompt_text) if final_prompt_text else 0
        answer_tokens = estimate_tokens(answer) if answer else 0
        generation_ms = 0.0
        if generation_start is not None and generation_end is not None:
            generation_ms = (generation_end - generation_start) * 1000.0
        performance_records.append({
            "question": question,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "prompt_tokens_est": int(prompt_tokens),
            "answer_tokens_est": int(answer_tokens)
        })
        print(f"完成：{question}")

    answer_payload = {"items": answered_items}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANSWER_PATH, 'w', encoding='utf-8') as f:
        json.dump(answer_payload, f, ensure_ascii=False, indent=2)
    performance_payload = {"items": performance_records}
    PERFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERFORMANCE_PATH, 'w', encoding='utf-8') as f:
        json.dump(performance_payload, f, ensure_ascii=False, indent=2)
    return {"answers": ANSWER_PATH, "performance": PERFORMANCE_PATH}


def run_pipeline() -> Dict[str, Path]:
    embedding = OpenAIEmbedding()
    vector_store = load_or_build_vector_store(embedding)
    output_paths = answer_questions(vector_store, embedding)
    return output_paths
