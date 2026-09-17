import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "models/qwen2.5-1.5b-summarizer-lora"

TEST_FILE = "data/test.jsonl"

MAX_NEW_TOKENS = 100

# Start small because we're running on CPU.
NUM_EXAMPLES = 20


SYSTEM_MESSAGE = (
    "You are a professional text summarization assistant. "
    "Produce a concise and factual summary of the provided document. "
    "Preserve important information and do not invent facts."
)


def load_test_data():
    examples = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    return examples[:NUM_EXAMPLES]


def create_prompt(tokenizer, document):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": f"Summarize the following document:\n\n{document}",
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_summary(model, tokenizer, document):
    prompt = create_prompt(tokenizer, document)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    summary = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return summary.strip()


def calculate_rouge(expected, generated):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = scorer.score(
        expected,
        generated,
    )

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def average_scores(scores):
    return {
        key: sum(item[key] for item in scores) / len(scores)
        for key in scores[0]
    }


def main():
    print("=" * 70)
    print("Qwen Summarizer Evaluation")
    print("=" * 70)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading test dataset...")

    examples = load_test_data()

    print(f"Testing {len(examples)} unseen examples")

    # -----------------------------------------------------
    # Base model
    # -----------------------------------------------------

    print("\nLoading base Qwen model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    base_model.eval()

    print("\nEvaluating BASE model...\n")

    base_scores = []

    base_outputs = []

    for index, example in enumerate(examples, start=1):
        generated = generate_summary(
            base_model,
            tokenizer,
            example["document"],
        )

        scores = calculate_rouge(
            example["summary"],
            generated,
        )

        base_scores.append(scores)
        base_outputs.append(generated)

        print(f"Base {index}/{len(examples)}")

    # Remove base model before loading another copy.
    del base_model

    # -----------------------------------------------------
    # Fine-tuned model
    # -----------------------------------------------------

    print("\nLoading Qwen again with LoRA adapter...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    print("\nEvaluating FINE-TUNED model...\n")

    fine_scores = []

    fine_outputs = []

    for index, example in enumerate(examples, start=1):
        generated = generate_summary(
            model,
            tokenizer,
            example["document"],
        )

        scores = calculate_rouge(
            example["summary"],
            generated,
        )

        fine_scores.append(scores)
        fine_outputs.append(generated)

        print(f"Fine-tuned {index}/{len(examples)}")

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    base_average = average_scores(base_scores)
    fine_average = average_scores(fine_scores)

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nBASE QWEN")
    print(f"ROUGE-1: {base_average['rouge1']:.4f}")
    print(f"ROUGE-2: {base_average['rouge2']:.4f}")
    print(f"ROUGE-L: {base_average['rougeL']:.4f}")

    print("\nFINE-TUNED QWEN")
    print(f"ROUGE-1: {fine_average['rouge1']:.4f}")
    print(f"ROUGE-2: {fine_average['rouge2']:.4f}")
    print(f"ROUGE-L: {fine_average['rougeL']:.4f}")

    print("\n")
    print("=" * 70)
    print("SAMPLE COMPARISONS")
    print("=" * 70)

    for i in range(min(5, len(examples))):
        print(f"\nExample {i + 1}")
        print("-" * 70)

        print("\nDOCUMENT:")
        print(examples[i]["document"])

        print("\nEXPECTED:")
        print(examples[i]["summary"])

        print("\nBASE QWEN:")
        print(base_outputs[i])

        print("\nFINE-TUNED QWEN:")
        print(fine_outputs[i])

        print()


if __name__ == "__main__":
    main()