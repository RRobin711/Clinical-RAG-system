#!/usr/bin/env python3
"""
Auto-verify gold QA answers using your LLM API.

For each question in gold_qa.json:
1. Reads the full extracted text for that doc_id
2. Sends it to Claude with the question
3. Extracts the specific answer with exact numbers
4. Writes the verified gold_qa.json back

Usage:
    python scripts/05_autofill_eval.py --eval_file evaluation/gold_qa.json --extracted_dir data/extracted
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anthropic


EXTRACTION_PROMPT = """You are reading a clinical psychology research paper. 
I need you to find the EXACT answer to this question from the text below.

QUESTION: {question}

RULES:
- Answer ONLY with specific facts, numbers, and values found in the text
- Include exact statistics: Cronbach's alpha values, ICC values, sensitivity/specificity percentages, cutoff scores, sample sizes, correlation coefficients
- If the paper reports multiple values, include all of them
- If the text does not contain the answer, respond with exactly: NOT_FOUND
- Keep the answer to 1-3 sentences maximum
- Do NOT make up or infer values that aren't explicitly stated

TEXT FROM PAPER:
{text}

ANSWER (specific facts and numbers only):"""


def autofill_answers(
    eval_file: Path,
    extracted_dir: Path,
    model: str = "claude-sonnet-4-20250514",
    delay: float = 1.0,
) -> None:
    # Load eval dataset
    items = json.loads(eval_file.read_text(encoding="utf-8"))
    client = anthropic.Anthropic()

    updated = 0
    not_found = 0
    skipped = 0
    errors = 0

    for i, item in enumerate(items):
        doc_id = item["source_doc_id"]
        question = item["question"]

        # Skip if already verified (no VERIFY prefix)
        if not item["answer"].startswith("VERIFY"):
            skipped += 1
            continue

        # Load the extracted text
        txt_path = extracted_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            print(f"[{i+1}/{len(items)}] SKIP - no text file for {doc_id}")
            skipped += 1
            continue

        text = txt_path.read_text(encoding="utf-8")

        # Truncate to ~80k chars if needed (API context limit)
        if len(text) > 80000:
            text = text[:80000] + "\n\n[TRUNCATED]"

        print(f"[{i+1}/{len(items)}] {question[:70]}...")
        print(f"         doc: {item.get('notes', doc_id)}")

        try:
            response = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(
                            question=question, text=text
                        ),
                    }
                ],
            )
            answer = response.content[0].text.strip()

            if answer == "NOT_FOUND" or "not found" in answer.lower()[:20]:
                print(f"         → NOT FOUND in paper")
                item["answer"] = f"NOT_FOUND: {item['answer']}"
                not_found += 1
            else:
                print(f"         → {answer[:100]}...")
                item["answer"] = answer
                updated += 1

        except Exception as e:
            print(f"         → ERROR: {e}")
            errors += 1

        # Rate limit
        time.sleep(delay)

    # Save updated dataset
    eval_file.write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print(f"  AUTOFILL COMPLETE")
    print(f"{'='*50}")
    print(f"  Updated:   {updated}")
    print(f"  Not found: {not_found}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(items)}")
    print(f"\n  Saved to: {eval_file}")
    print(f"\n  NEXT: Review the answers, especially NOT_FOUND ones.")
    print(f"  Then run: python scripts/05_evaluate.py --eval_file {eval_file} --no_wandb")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-verify gold QA answers using Claude"
    )
    parser.add_argument(
        "--eval_file", type=Path, default=Path("evaluation/gold_qa.json")
    )
    parser.add_argument(
        "--extracted_dir", type=Path, default=Path("data/extracted")
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between API calls (rate limiting)",
    )
    args = parser.parse_args()
    autofill_answers(
        args.eval_file, args.extracted_dir, model=args.model, delay=args.delay
    )


if __name__ == "__main__":
    main()
