import os
import re
import gc
import torch
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
token = ''
# Set PyTorch memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

class RecipeFineTuner:
    def __init__(
        self,
        base_model_name="meta-llama/Llama-3.2-3B",
        dataset_path=None,
        output_dir="/recipe_model/"
    ):
        self.base_model_name = base_model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Even more aggressive BitsAndBytes config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Minimal LoRA config
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", "v_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Modified training arguments to remove evaluation-related settings
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=50,
            save_strategy="epoch",
            max_grad_norm=0.3,
            gradient_checkpointing=True,
            warmup_ratio=0.05,
            weight_decay=0.01,
            optim="paged_adamw_8bit",
            group_by_length=True,
            report_to="none",
            lr_scheduler_type="cosine",
            max_steps=-1,
            save_total_limit=1,  # Keep only the last checkpoint
            ddp_find_unused_parameters=False,
            torch_compile=False,  # Disable torch.compile to save memory
        )

    def prepare_model(self):
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
            
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            token=token,
            model_max_length=1200,
            trust_remote_code=True
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            token=token,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )

        if hasattr(model.config, 'rope_scaling'):
            delattr(model.config, 'rope_scaling')

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        if hasattr(model.config, "use_memory_efficient_attention"):
            model.config.use_memory_efficient_attention = True

        return model, tokenizer

    def prepare_data(self):
        print("Loading dataset...")
        data = pd.read_csv(
            self.dataset_path,
            usecols=['TranslatedRecipeName', 'Cleaned-Ingredients', 'TranslatedInstructions']
        )
        
        print("Processing ingredients...")
        data['Processed_Ingredients'] = data['Cleaned-Ingredients'].apply(
            lambda x: ", ".join([item.strip() for item in re.sub(r'\s*\(.*?\)\s*', '', x).split(',')])
        )

        train_data = data

        def format_recipe(row):
            return f"""<s>[INST]
Ingredients:
{row['Cleaned-Ingredients']}[/INST]
Instructions:
{row['TranslatedInstructions']}</s>"""

        print("Formatting recipes...")
        train_texts = [format_recipe(row) for _, row in train_data.iterrows()]

        del data, train_data
        gc.collect()

        return Dataset.from_dict({"text": train_texts}), 

    def train(self):
        train_dataset, = self.prepare_data()
        model, tokenizer = self.prepare_model()

        def tokenize(examples):
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors=None
            )
            return outputs

        print("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            tokenize,
            remove_columns=train_dataset.column_names,
            batched=True,
            batch_size=1,
            desc="Tokenizing train data"
        )

        del train_dataset
        gc.collect()
        torch.cuda.empty_cache()

        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_train,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

        print("Starting training...")
        trainer.train()

        final_save_dir = os.path.join(self.output_dir, "final_model")
        trainer.save_model(final_save_dir)
        tokenizer.save_pretrained(final_save_dir)

        return final_save_dir

def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    
    dataset_path = 'artifacts/Cleaned_Indian_Food_Dataset.csv'
    output_dir = 'recipe_model/'

    fine_tuner = RecipeFineTuner(
        dataset_path=dataset_path,
        output_dir=output_dir
    )

    try:
        model_save_dir = fine_tuner.train()
        print(f"Training completed. Model saved at {model_save_dir}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU out of memory. Consider further reducing dataset size or sequence length.")
            print("Current GPU memory usage:")
            print(torch.cuda.memory_summary())
        raise e

if __name__ == "__main__":
    main()