from transformers import AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import torch
from PIL import Image
from transformers import pipeline
from datasets import load_dataset
import requests
from PIL import Image
import random
import pandas as pd
import re
import json
import csv
import io
import base64


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

def generate_response(input_text,image):
    """
    Genera una risposta utilizzando un modello LLaVA 
    """

    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # Carica il modello multimodale
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cpu().eval()


    # Prepara la conversazione
    conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{input_text}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
    ]
    
    # Carica le immagini
    pil_images = load_pil_images(conversation)

    # Prepara gli input per il modello
    prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # Ottieni gli embeddings per l'immagine
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Esegui il modello per ottenere la risposta
    outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=2000, 
    do_sample=False, 
    use_cache=True
    )

    # Decodifica la risposta
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    # Aggiungi messages se desiderato
    messages = {"content": answer, "role": "<|Assistant|>"}
    # print(f"{prepare_inputs['sft_format'][0]}", answer)
    return answer, messages

def run_Benchmark(dataset,benchmark,startnum):
    """
    Esegue un benchmark su un dataset utilizzando un modello LLaVA e salva i risultati in un file csv.
    """
    # # Inizializzare la pipeline del modello LLaVA
    # pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

    # File di output csv
    output_file = f'evaluation_results_{benchmark}.csv'
    results = []  # Lista per accumulare i risultati

    # Estrarre dati dal file Excel
    # dataframe = ExtractDataExcel(file_path)
    dataframe=pd.DataFrame(dataset)
    dataframe_temp = dataframe.iloc[startnum:]

    original_prompt="You will be shown an image, you will have to answer the question related to the image."

    # image_url="https://img.freepik.com/free-photo/abstract-surface-textures-white-concrete-stone-wall_74190-8189.jpg" #Blank image
    i=1

    for index, row in dataframe_temp.iterrows():
        image_url = "images/image{}.jpg".format(i)
        input_text = row['question']
        i=i+1
        
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
            "image":image_url
        }
        results.append(result)
        
        print(f'Processed prompt {index + 1}/{len(dataframe_temp)}')

    # Scrivere i risultati nel file JSON
    with open(output_file, 'w',newline='') as csvfile:
        fieldnames=['original_prompt','optimized_prompt','optimized_prompt_ExP_CoT','answer_optimized_Exp_CoT','messages_optimized_Exp_CoT','answer_original','answer_optimized','original_messages','optimized_messages','image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark completato. Risultati salvati in {output_file}.")
    return


dataframe_testmini=pd.read_csv('Datasets/algebra_testmini.csv')

run_Benchmark(dataframe_testmini,'algebra_testmini_JANUS', 0)






