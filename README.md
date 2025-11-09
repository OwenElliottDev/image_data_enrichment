# Image Enrichment

A Rust CLI tool to batch-process images through an Ollama model, generating JSON outputs containing captions or structured data. Supports optional JSON schemas, batching, and skipping existing outputs.

## Features

* Batch image processing with configurable batch size.
* Supports JPG, PNG, BMP, GIF, and WEBP images.
* Converts images to Base64 and sends them to an Ollama model.
* Optional JSON schema validation for structured responses.
* Skips already processed images to avoid duplicates.
* Pretty-printed JSON output support.
* Progress bar display for batch processing.

## Installation

```bash
cargo install --path .
```

## Usage

```bash
image_enrichment --dir <INPUT_DIR> --api_url <OLLAMA_API_URL> --model <MODEL_NAME> [OPTIONS]
```

## Arguments

| Argument          | Description                                    | Default                            |
| ----------------- | ---------------------------------------------- | ---------------------------------- |
| `--dir`           | Directory containing input images              | **Required**                       |
| `--api_url`       | Ollama API URL                                 | **Required**                       |
| `--model`         | Ollama model name                              | **Required**                       |
| `--schema`        | Path to JSON schema file for structured output | Optional                           |
| `--prompt`        | Prompt to send to the model                    | `"What do you see in this image?"` |
| `--output_dir`    | Directory to save output JSON files            | Same as input directory            |
| `--debug`         | Enable verbose debug logging                   | `false`                            |
| `--options`       | JSON string of additional model options        | Optional                           |
| `--pretty-json`   | Pretty-format the JSON output                  | `false`                            |
| `--batch-size`    | Number of images per batch                     | `1`                                |
| `--skip-existing` | Skip images that already have output JSON      | `false`                            |
| `--suffix`        | Suffix to append to output JSON filenames      | `""`                               |