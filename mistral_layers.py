import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

def load_model(model_name: str = MODEL_NAME, local_dir: str | None = None):
    """Load tokenizer and model from Hugging Face or a local directory."""
    path = local_dir if local_dir is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_dir is not None)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=local_dir is not None,
    )
    return tokenizer, model

def print_layers(model):
    """Print a summary of each transformer layer."""
    for idx, layer in enumerate(model.model.layers):
        cls_name = layer.__class__.__name__
        print(f"Layer {idx}: {cls_name}")


def generate_example(tokenizer, model, prompt: str):
    """Generate text for a simple prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inspect Mistral model layers")
    parser.add_argument("--model-dir", help="Path to local weights", default=None)
    args = parser.parse_args()

    tokenizer, model = load_model(local_dir=args.model_dir)
    print_layers(model)
    example = generate_example(tokenizer, model, "Hello, world!")
    print("\nSample generation:\n", example)


if __name__ == "__main__":
    main()
