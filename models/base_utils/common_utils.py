import yaml
import re
import json
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