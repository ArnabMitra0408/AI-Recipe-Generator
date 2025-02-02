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

- **Recall**: Evaluates the completeness of the generated recipe in incorporating input ingredients.

  **Formula:**
  ```markdown
  Recall = (Relevant ingredients used) / (Total input ingredients provided)

- **BLEU Score**: Assesses the linguistic similarity between generated recipes and human-written recipes using n-gram overlap.

  **Formula:**
  ```markdown
  BLEU = BP * exp(sum(w_n * log P_n))
  Where:

     1. BP is the brevity penalty
     2. P_n is the modified precision for n-grams

## Base Model Performance

The baseline results before fine-tuning are as follows:

| Model        | Prompt | Precision | Recall | BLEU Score |
|--------------|--------|-----------|--------|------------|
| LLAMA2       | 1      | 0.88      | 0.85   | 0.23       |
| LLAMA3.2     | 1      | 0.93      | 0.94   | 0.34       |
| Mistral-7B   | 1      | 0.90      | 0.87   | 0.36       |

## Fine-Tuned Model Performance

Fine-tuned results for LLAMA3.2 using QLoRA are as follows:

| Prompt | Precision | Recall | BLEU Score |
|--------|-----------|--------|------------|
| 1      | 0.96      | 0.96   | 0.52       |
| 2      | 0.95      | 0.94   | 0.43       |
| 3      | 0.95      | 0.95   | 0.47       |

### Key Observations:
- Fine-tuning significantly improved the model’s ability to generate accurate and culturally authentic recipes.
- Prompt 1 yielded the best overall performance in terms of precision, recall, and BLEU score.

## Conclusion

This project successfully demonstrated the effectiveness of fine-tuning LLMs for recipe generation tasks. The results validate the use of QLoRA as a resource-efficient method to enhance model performance. Future work will focus on:

- Expanding the dataset to improve diversity and accuracy.
- Refining prompt engineering techniques.
- Integrating user feedback to continuously optimize performance.

## Future Work

- **Expand Dataset**: Increase the size and diversity of the dataset for broader recipe generation.
- **Refine Prompt Engineering**: Experiment with different prompt structures for further improvements in model performance.
- **User Feedback**: Incorporate feedback from end-users to fine-tune the model further and address real-world needs.

Feel free to explore the code and contribute to this project!
