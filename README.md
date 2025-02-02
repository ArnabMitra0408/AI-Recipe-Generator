# AI-Recipe-Generator

## Project Overview

This project explores the use of Large Language Models (LLMs) for AI-driven recipe generation based on input ingredients. The core objective is to fine-tune LLMs to generate culturally authentic and contextually relevant Indian recipes. Fine-tuning is performed using Quantized Low-Rank Adaptation (QLoRA) to optimize model performance efficiently. The models are evaluated using precision, recall, and BLEU score to assess their ability to generate high-quality recipes.

## Data Used

The dataset consists of **5,900 different Indian recipes** with structured metadata, including:

- **Ingredients**
- **Instructions**

Data preprocessing included tokenization, standardization of ingredient names, and removal of duplicates to enhance model training quality.

## Prompts Used

Three different prompt structures were tested for recipe generation, with Prompt 1 yielding the best results in terms of precision, recall, and BLEU score.

## Models Used

Three state-of-the-art LLMs were selected for evaluation:

- **LLAMA2**
- **LLAMA3.2**
- **Mistral-7B**

## Fine-Tuning Method Used

Fine-tuning was performed using **QLoRA (Quantized Low-Rank Adaptation)** to optimize the models with minimal computational overhead. This involved:

- Injecting low-rank matrices into model layers
- Using 8-bit quantization for efficiency
- Training with adaptive learning rates

## Metrics Used

Three key metrics were used to evaluate model performance:

- **Precision**: Measures how accurately the generated recipe uses relevant ingredients.

  **Formula:**
  ```markdown
  Precision = (Relevant ingredients used) / (Total ingredients used in the generated recipe)
