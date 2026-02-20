from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .llm import LLMResult, chat_completion
from .prompting import build_grounded_messages
from .retriever import RetrievedChunk, Retriever
from .settings import Settings, load_env


@dataclass(frozen=True)
class AnswerWithSources:
    answer: str
    model: str
    retrieved: List[RetrievedChunk]


def answer_question(
    *,
    index_dir: Path,
    question: str,
    k: int = 5,
) -> AnswerWithSources:
    """
    End-to-end: query -> retrieve -> prompt -> LLM.
    """
    load_env()
    settings = Settings.from_env()
    
    retriever = Retriever(index_dir=index_dir)
    retrieved = retriever.retrieve(question, k=k)
    
    messages = build_grounded_messages(question=question, retrieved=retrieved)
    llm_res: LLMResult = chat_completion(messages=messages, settings=settings)
    
    return AnswerWithSources(answer=llm_res.text, model=llm_res.model, retrieved=retrieved)