# faster-reasoning

This repository includes a small example showing how to load the open-source
[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) language
model with the Hugging Face `transformers` library. The script `mistral_layers.py`
loads the model, prints a summary of each transformer layer, and generates a
short sample response.

## Requirements

- Python 3.8+
- `torch`
- `transformers`

Install dependencies with:

```bash
pip install torch transformers
```

## Usage

Run the script to view the model layers and a sample generation:

```bash
python3 mistral_layers.py
```

Loading the model requires downloading the weights from Hugging Face on the
first run. Ensure you have internet access and enough disk space (~15GB).

## Custom architecture implementation

The file `mistral_full.py` provides a minimal PyTorch implementation of the
Mistral model. It mirrors the architecture used by the released 7B weights so
you can experiment with changing individual layers. The helper `load_pretrained`
function loads the official weights from Hugging Face into this custom model.

To print a short summary of each layer run:

```bash
python3 mistral_full.py
```

The example relies on the same `torch` and `transformers` dependencies listed
above.
