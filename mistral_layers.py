import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

def load_model(model_name: str = MODEL_NAME):
    """Load tokenizer and model from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
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
    tokenizer, model = load_model()
    print_layers(model)
    example = generate_example(tokenizer, model, "Hello, world!")
    print("\nSample generation:\n", example)


if __name__ == "__main__":
    main()
