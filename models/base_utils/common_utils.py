import yaml
import re
import json
from nltk.translate.bleu_score import sentence_bleu



def read_params(config_path:str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def clean_indian_ingredients(ingredient_string):
    cleaned_string = re.sub(r'\s*\(.*?\)\s*', '', ingredient_string)
    return [item.strip() for item in cleaned_string.split(',')]

def calculate_precision(actual_ingredients,gen_ingredients):
    total_ingredients_used = len(gen_ingredients)
    relevant_ingredients_used = len(set(actual_ingredients).intersection(set(gen_ingredients)))
    precision = relevant_ingredients_used / total_ingredients_used
    return precision

def calculate_recall(actual_ingredients,gen_ingredients):
    total_input_ingredients = len(actual_ingredients)
    relevant_ingredients_used = len(set(actual_ingredients).intersection(set(gen_ingredients)))
    recall = relevant_ingredients_used / total_input_ingredients
    return recall 

def write_json(file_path,data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def calculate_bleu(reference_text,candidate_text):
    reference_tokens = reference_text.split('/n')
    candidate_text='\n '.join(candidate_text)
    candidate_tokens = candidate_text
    weights = (0.5, 0.5)
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens,weights=weights)
    return bleu_score

def train_test_split(df,test_size:int=100):
    if test_size > len(df):
        raise ValueError("Test size cannot be larger than the number of rows in the DataFrame.")
    
    # Split the DataFrame
    train_set = df.iloc[:-test_size]
    test_set = df.iloc[-test_size:]
    
    return train_set, test_set