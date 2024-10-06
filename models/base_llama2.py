from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import re
from base_utils.common_utils import write_json,read_params,clean_indian_ingredients,calculate_precision,calculate_recall,train_test_split
from prompts.prompts import prompts


def create_chain(model_name,prompt):
    llm = Ollama(model=model_name)
    output_parser=StrOutputParser(
    )
    chain=prompt|llm|output_parser
    return chain

def parse_response(response:str):
    lines = response.split('\n')
    gen_ingredients = []
    gen_instructions = []
    in_ingredients = False
    in_instructions = False
    for line in lines:
        line = line.strip() 
        if line == "Ingredients Used:":
            in_ingredients = True
            continue
        elif line == "Instructions:":
            in_ingredients = False
            in_instructions = True
            continue
        if in_ingredients and line.startswith("*"):
            gen_ingredients.append(line[1:].strip())
        elif in_instructions and line.startswith("*"):
            gen_instructions.append(line[1:].strip())   
    return gen_ingredients,gen_instructions 
    
def metrics(data:pd.DataFrame,model_name:str,out_path:str,prompt_out_path:str,num_responses:int=1)->None:
    all_prompts=prompts()
    i=0
    all_metrics=[]
    prompt_responses=[]
    for prompt in all_prompts:
        print(prompt)
        responses={}
        precision_all=[]
        recall_all=[]
        for row in range(data.shape[0]):
            print(f'Entry:{row}')
            actual_ingredients=data['Processed_Ingredients'][row]
            chain=create_chain(model_name,prompt)
            response=chain.invoke({'Ingredients':actual_ingredients})
            gen_ingredients,gen_instructions=parse_response(response)
            gen_ingredients = [ingredient.lower() for ingredient in gen_ingredients]
            precision=calculate_precision(actual_ingredients,gen_ingredients)
            recall=calculate_recall(actual_ingredients,gen_ingredients)
            precision_all.append(precision)
            recall_all.append(recall)
            responses[row]=response
        avg_precision=sum(precision_all)/len(precision_all)
        avg_recall=sum(recall_all)/len(recall_all)
        print(f'Precision is: {avg_precision}')   
        print(f'Recall is: {avg_recall}')
        i+=1
        meta_data={'Prompt_Number':i,'Precision':avg_precision,'Recall':avg_recall}
        prompt_responses.append(responses)
        all_metrics.append(meta_data)
    write_json(out_path,all_metrics)
    write_json(prompt_out_path,prompt_responses)
    return

if __name__=='__main__':
    args=ArgumentParser()
    args.add_argument("--config_path",'-c',default='params.yaml')
    parsed_args=args.parse_args()
    configs=read_params(parsed_args.config_path)
    num_responses=int(parsed_args.num_responses)
    output_path=configs['data_path']['llama2_base']
    prompt_out_path=configs['data_path']['llama2_base_prompts']
    test_size=configs['test_size']['test_size']
    print("LLAMA2 is Running")
    indian_data_path=configs['data_dir']['indian_data']

    data=pd.read_csv(indian_data_path,
                     usecols=['TranslatedRecipeName',
                              'Cleaned-Ingredients','TranslatedInstructions','Ingredient-count'])
    data['Processed_Ingredients'] = data['Cleaned-Ingredients'].apply(clean_indian_ingredients)
    
    model_name=configs['model_names']['llama2']
    
    #Test_data
    _,test_data=train_test_split(data,test_size)
    test_data.reset_index(inplace=True,drop=True)
    metrics(test_data,model_name,output_path,prompt_out_path,num_responses)
    print("Program Executed Successfully")
