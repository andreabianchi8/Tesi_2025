from transformers import pipeline
from datasets import load_dataset
import requests
from PIL import Image
import random
import csv
import pandas as pd
import re
import json
import csv


def prompt_optimization(sample_prompt, input_text):
    """
    Ottimizza il prompt originale utilizzando una serie di tecniche descritte.
    """
    optimization_prompt = f'''
Your available prompting techniques include, but are not limited to the following:
- Crafting an expert who is an expert at the given task, by writing a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective.
- Explaining step-by-step how the problem should be tackled, and making sure the model explains step-by-step how it came to the answer. You can do this by adding "Let's think step-by-step".
- Imagining three different experts who are discussing the problem at hand. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.
- Making sure all information needed is in the prompt, adding where necessary but making sure the question remains having the same objective.

Output instructions:
You should ONLY return the reformulated prompt. Make sure to include ALL information from the given prompt to reformulate.
'''
    # Uniamo il prompt ottimizzato con l'input specifico
    optimized_prompt = f"{optimization_prompt}\n\nPrompt to optimize:\n{sample_prompt}\n\nInput for question:\n\"\"\"\n{input_text}\n\"\"\""
    return optimized_prompt

def prompt_optimization_ExP_CoT(sample_prompt, input_text):
    """
    Ottimizza il prompt originale utilizzando una serie di tecniche descritte.
    """
    optimization_prompt = f'''
Your available prompting techniques include Expert Prompting and Chain of Thought, use a combination of both:
- Crafting an expert who is an expert at the given task, by writing a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective.
- Explaining step-by-step how the problem should be tackled, and making sure the model explains step-by-step how it came to the answer. You can do this by adding "Let's think step-by-step".
- Making sure all information needed is in the prompt, adding where necessary but making sure the question remains having the same objective.

Output instructions:
You should ONLY return the reformulated prompt. Make sure to include ALL information from the given prompt to reformulate.
'''
    # Uniamo il prompt ottimizzato con l'input specifico
    optimized_prompt = f"{optimization_prompt}\n\nPrompt to optimize:\n{sample_prompt}\n\nInput for question:\n\"\"\"\n{input_text}\n\"\"\""
    return optimized_prompt

def generate_response(pipe,input_text,image_url):
    """
    Genera una risposta utilizzando un modello LLaVA 
    """
    messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "url": image_url},
          {"type": "text", "text": input_text},
        ],
    },
  ]
    
    # Chiamata al modello
    response = pipe(text=messages,max_new_tokens=2000,do_sample=False)

    # Estrai il testo generato
    generated_response = response[0]["generated_text"]
    return generated_response, messages

def ExtractDataExcel(file_path):
    """
    Legge un file Excel e restituisce un DataFrame.
    """
    df = pd.read_excel(file_path)
    df.rename(columns={
        'Input': 'input',
        'Target': 'target',
        'Prompt': 'user_prompt',
        'Example_Output': 'Example_Output'
    }, inplace=True)
    return df


def run_Benchmark(dataset,benchmark,startnum):
    """
    Esegue un benchmark su un dataset utilizzando un modello LLaVA e salva i risultati in un file csv.
    """
    # Inizializzare la pipeline del modello LLaVA
    pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

    # File di output csv
    output_file = f'evaluation_results_{benchmark}.csv'
    results = []  # Lista per accumulare i risultati

    # Estrarre dati dal file Excel
    # dataframe = ExtractDataExcel(file_path)

    dataframe = dataset

    original_prompt="You will be shown an image, you will have to answer the question related to the image."

    # image_url="https://img.freepik.com/free-photo/abstract-surface-textures-white-concrete-stone-wall_74190-8189.jpg" #Blank image
    index=0
    for key, value in dataframe.items():
        if "inductive" in value['skill'] or "numerical" in value['skill']:
            image_url="./images/{}".format(value['imagename'])
            input_text = value['question']
            ground_truth=value['answer']
            reasoning=value['reasoning']
        
        
            # Ottimizzazione del prompt
            optimized_prompt = prompt_optimization(original_prompt, input_text)
            optimized_prompt_ExP_CoT=prompt_optimization_ExP_CoT(original_prompt,input_text)

            # Generare risposte per il prompt originale e quello ottimizzato
            answer_original, messages_original = generate_response(original_prompt+input_text,image_url)
            answer_optimized, messages_optimized = generate_response(optimized_prompt,image_url)
            answer_optimized_Exp_CoT,messages_optimized_Exp_CoT= generate_response(optimized_prompt_ExP_CoT,image_url)

            # Accumulare i risultati in un dizionario
            result = {
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "optimized_prompt_ExP_CoT": optimized_prompt_ExP_CoT,
            "answer_optimized_Exp_CoT": answer_optimized_Exp_CoT,
            "messages_optimized_Exp_CoT": messages_optimized_Exp_CoT,
            "answer_original": answer_original,
            "answer_optimized": answer_optimized,
            "original_messages": messages_original,
            "optimized_messages": messages_optimized,
            'ground_truth': ground_truth,
            'reasoning':reasoning,
            "image":image_url
            }
            results.append(result)
        
            print(f'Processed prompt {index + 1}/{len(dataframe)}')
            index=index+1

    # Scrivere i risultati nel file JSON
    with open(output_file, 'w',newline='') as csvfile:
        fieldnames=['original_prompt','optimized_prompt','optimized_prompt_ExP_CoT','answer_optimized_Exp_CoT','messages_optimized_Exp_CoT','answer_original','answer_optimized','original_messages','optimized_messages','ground_truth','reasoning','image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark completato. Risultati salvati in {output_file}.")
    return

dataframe_testmini=pd.read_csv('algebra_testmini.csv')

run_Benchmark(dataframe_testmini,'algebra_testmini_LLava', 0)