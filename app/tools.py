from transformers import pipeline

# Stable text-generation model
generator = pipeline(
    "text-generation",
    model="distilgpt2"   # smaller & more stable than gpt2
)

def direct_llm(prompt: str) -> str:
    """
    Generate clean answer using text-generation model.
    """

    result = generator(
        prompt,
        max_new_tokens=80,
        truncation=True,
        do_sample=False  # makes output deterministic
    )

    generated_text = result[0]["generated_text"]

    # Remove prompt part from response
    if generated_text.startswith(prompt):
        cleaned = generated_text[len(prompt):].strip()
    else:
        cleaned = generated_text.strip()

    # Cut off overly long junk
    cleaned = cleaned.split("\n")[0]

    return cleaned