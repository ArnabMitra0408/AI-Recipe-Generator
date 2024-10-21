from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import re
from base_utils.common_utils import write_json,read_params,clean_indian_ingredients,calculate_precision,calculate_recall,calculate_bleu,train_test_split
from prompts.prompts import prompts


def create_chain(model_name,prompt):
    llm = Ollama(model=model_name)
    chain=prompt | llm | StrOutputParser()
    return chain

def parse_response(response:str):
    lines = response.split('\n')
    gen_ingredients = []
    gen_instructions = []
    current_section = None
    for line in lines:
        stripped_line = line.strip()  
    
        if stripped_line == 'Ingredients Used:':
            current_section = 'ingredients'
            continue  
        elif stripped_line == 'Instructions:':
            current_section = 'instructions'
            continue 
        elif stripped_line.startswith('- '):
            # Extract the item after '- '
            item = stripped_line[2:].strip()
            if current_section == 'ingredients':
                gen_ingredients.append(item)
            elif current_section == 'instructions':
                gen_instructions.append(item)
        else:
            continue   
    return gen_ingredients,gen_instructions 
    
def metrics(data:pd.DataFrame,model_name:str,metrics_out_path:str,prompt_out_path:str)->None:
    all_prompts=prompts()
    i=0
    all_metrics=[]
    prompt_responses=[]
    for prompt in all_prompts:
        print(prompt)
        responses={}
        precision_all=[]
        recall_all=[]
        bleu_all=[]
        chain=create_chain(model_name,prompt)
        for row in range(data.shape[0]):
            print(f'Entry:{row}')
            actual_ingredients=data['Processed_Ingredients'][row]
            actual_instructions=data['TranslatedInstructions'][row]
            response=chain.invoke({'Ingredients':actual_ingredients})
            gen_ingredients,gen_instructions=parse_response(response)

            gen_ingredients = [ingredient.lower() for ingredient in gen_ingredients]
            precision=calculate_precision(actual_ingredients,gen_ingredients)
            recall=calculate_recall(actual_ingredients,gen_ingredients)
            bleu=calculate_bleu(actual_instructions, gen_instructions)
            precision_all.append(precision)
            recall_all.append(recall)
            bleu_all.append(bleu)
            responses[row]=response
        avg_precision=sum(precision_all)/len(precision_all)
        avg_recall=sum(recall_all)/len(recall_all)
        avg_bleu=sum(bleu_all)/len(bleu_all)
        print(f'Precision is: {avg_precision}')   
        print(f'Recall is: {avg_recall}')
        print(f"Bleu is: {avg_bleu:.4f}")
        i+=1
        meta_data={'Prompt_Number':i,'Precision':avg_precision,'Recall':avg_recall, "BLEU":avg_bleu}
        prompt_responses.append(responses)
        all_metrics.append(meta_data)
    write_json(metrics_out_path,all_metrics)
    write_json(prompt_out_path,prompt_responses)
    return

if __name__=='__main__':
    args=ArgumentParser()
    args.add_argument("--config_path",'-c',default='params.yaml')
    parsed_args=args.parse_args()
    configs=read_params(parsed_args.config_path)
    metrics_out_path=configs['data_path']['llama3_base']
    prompt_out_path=configs['data_path']['llama3_base_prompts']
    test_size=int(configs['test_size']['test_size'])
    print("LLAMA3.2 is Running")
    indian_data_path=configs['data_dir']['indian_data']

    data=pd.read_csv(indian_data_path,
                     usecols=['TranslatedRecipeName',
                              'Cleaned-Ingredients','TranslatedInstructions','Ingredient-count'])
    data['Processed_Ingredients'] = data['Cleaned-Ingredients'].apply(clean_indian_ingredients)
    
    model_name=configs['model_names']['llama3']
    
    #Test_data
    _,test_data=train_test_split(data,test_size)
    test_data.reset_index(inplace=True,drop=True)
    metrics(test_data,model_name,metrics_out_path,prompt_out_path)
    print("Program Executed Successfully")
