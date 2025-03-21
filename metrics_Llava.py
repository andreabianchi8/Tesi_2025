import pandas as pd
import asyncio
import re
from ragas import SingleTurnSample
from ragas.metrics import BleuScore
from ragas.metrics import RougeScore
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
import pickle
import json
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

import os
ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")  # Usa una variabile d'ambiente


model = SentenceTransformer('all-MiniLM-L6-v2')

# Carica il dataset dell'evaluation results Llava
dataset_evaluation=pd.read_csv("evaluation_results_LogicVista_Llava.csv")

# Caricare il dataset JSON LogicVista
with open("Datasets/LogicVista_dataset.json", "r", encoding="utf-8") as file:
    dataset_reference = json.load(file)

# Dizionario per salvare i risultati delle risposte: originali,ottimizzate Exp Cot, ottimizzate Exp Cot ToT
results = {
    'answer_optimized_Exp_CoT': [],
    'answer_original': [],
    'answer_optimized': []
}

bleu_score_answer_exp_cot=[]
bleu_score_answer_original=[]
bleu_score_answer_optimized=[]

for index, row in dataset_evaluation.iterrows():
    for column in results.keys():
        
        answer_text = row[column]

        results[column].append(answer_text)


# Ora i risultati sono in liste separate
results_answer_Exp_CoT = results['answer_optimized_Exp_CoT']
results_answer_original = results['answer_original']
results_answer_optimized = results['answer_optimized']

################################################### Verifica se contiene l'answer [0 No, 1 Sì] [Correttezza]

# Carica il modello Llama 3 (8B) da Hugging Face
llm = pipeline("text-generation", 
               model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
               token=ACCESS_TOKEN)

def preprocess_text(text):
    """Rende il testo più uniforme rimuovendo punteggiatura e normalizzando gli spazi."""
    text = text.lower().strip()
    # text = re.sub(r'[^a-z0-9 ]', '', text)  # Rimuove caratteri speciali
    return text

def check_ground_truth_in_response(ground_truth, model_response,type,threshold=90):
    """Verifica se la risposta ha lo stesso significato della ground truth."""
    
    ground_truth=preprocess_text(ground_truth)
    model_response=preprocess_text(model_response
                                   )
    print(f"GT({type}): {ground_truth} ")
    print(f"Model Response({type}): {model_response}")
    print("\n")

    prompt = f"""
    Analizza il seguente testo per determinare se il modello ha fornito la risposta corretta. 
    Non limitarti a cercare la lettera corretta nella risposta, ma verifica se il significato della risposta 
    implica chiaramente l'opzione corretta.

    Risposta generata: "{model_response}"
    Opzione corretta: "{ground_truth}"

    Restituisci solo "1" se la risposta implica chiaramente l'opzione corretta, 
    oppure "0" se non la implica. Non fornire altre spiegazioni.
    """
    result = llm(prompt, max_new_tokens=10)[0]["generated_text"]

    print("Result: ",result)  # Deve restituire "1" o "0"
    return 1 if "1" in result else 0
    

results_correctness_expt_cot=[]
results_correctness_answer_original=[]
results_correctness_answer_optimized=[]
index=0

for key, value in dataset_reference.items(): 
        if "inductive" in value['skill'] or "numerical" in value['skill']:
            results_correctness_expt_cot.append(check_ground_truth_in_response(value['answer'],results_answer_Exp_CoT[index],'Exp_Cot'))
            results_correctness_answer_original.append(check_ground_truth_in_response(value['answer'],results_answer_original[index],'original'))
            results_correctness_answer_optimized.append(check_ground_truth_in_response(value['answer'],results_answer_optimized[index],'optimized'))
            index=index+1



print("Correttezza risposte ottimizzate con Expert Prompting e Chain of Thought:",len(results_correctness_expt_cot))
print(results_correctness_expt_cot)
print("Correttezza risposte originali senza ottimizzazione:",len(results_correctness_answer_original))
print(results_correctness_answer_original)
print("Correttezza risposte ottimizzate (Expert Prompting,ToT,CoT):",len(results_correctness_answer_optimized))
print(results_correctness_answer_optimized)

# Salva le liste in un file pickle
with open("pickle_data/results_correctness_expt_cot_llava.pkl", "wb") as f:
    pickle.dump(results_correctness_expt_cot, f)

with open("pickle_data/results_correctness_answer_original_llava.pkl", "wb") as f:
    pickle.dump(results_correctness_answer_original, f)

with open("pickle_data/results_correctness_answer_optimized_llava.pkl", "wb") as f:
    pickle.dump(results_correctness_answer_optimized, f)

##################################################### Accuracy in % (numero di 1 / totale degli elementi) x 100 

print("Accuracy '%' risposte ottimizzate con Expert Prompting e Chain of Thought:")
accuracy_expt_cot=(sum(results_correctness_expt_cot) / len (results_correctness_expt_cot)) * 100
print(accuracy_expt_cot)

print("Accuracy '%' risposte originali senza ottimizzazione:")
accuracy_answer_original=(sum(results_correctness_answer_original) / len (results_correctness_answer_original)) * 100
print(accuracy_answer_original)

print("Accuracy '%' risposte ottimizzate (Expert Prompting,ToT,CoT):")
accuracy_answer_optimized=(sum(results_correctness_answer_optimized) / len (results_correctness_answer_optimized)) * 100
print(accuracy_answer_optimized)

# Salva le liste in un file pickle
with open("pickle_data/accuracy_expt_cot_llava.pkl", "wb") as f:
    pickle.dump(accuracy_expt_cot, f)

with open("pickle_data/accuracy_answer_original_llava.pkl", "wb") as f:
    pickle.dump(accuracy_answer_original, f)

with open("pickle_data/accuracy_answer_optimized_llava.pkl", "wb") as f:
    pickle.dump(accuracy_answer_optimized, f)

##################################################### Semantic similarity

# Wrapper per rendere SentenceTransformer asincrono
class AsyncSentenceTransformer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    async def aembed_documents(self, texts):
        # Esegui l'encoding in un thread separato per renderlo asincrono
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.model.encode, texts)
        return embeddings

# Funzione per calcolare la similarità semantica
async def semantic_similarity(result_answer, evaluator_embedding):
    scores = []
    index=0

    for key, value in dataset_reference.items():
        if "inductive" in value['skill'] or "numerical" in value['skill']:
            sample = SingleTurnSample(
            response=preprocess_text(result_answer[index]), 
            reference=preprocess_text(value['reasoning'])
            )
            scorer = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(evaluator_embedding))
            score = await scorer.single_turn_ascore(sample)
            scores.append(score)
            index=index+1
        #print(f"Semantic similarity score: {score}")

    return scores

# Creazione del wrapper asincrono per SentenceTransformer
evaluator_embedding = AsyncSentenceTransformer('paraphrase-MiniLM-L6-v2')

# Esegui le valutazioni per le risposte
results_semantic_sim_expt_cot = asyncio.run(semantic_similarity(results_answer_Exp_CoT, evaluator_embedding))
results_semantic_sim_answer_original = asyncio.run(semantic_similarity(results_answer_original, evaluator_embedding))
results_semantic_sim_answer_optimized = asyncio.run(semantic_similarity(results_answer_optimized, evaluator_embedding))

print("Semantic Similarity risposte ottimizzate con Expert Prompting e Chain of Thought:")
print(results_semantic_sim_expt_cot)
print("Semantic Similarity risposte originali senza ottimizzazione:")
print(results_semantic_sim_answer_original)
print("Semantic Similarity risposte ottimizzate (Expert Prompting,ToT,CoT):")
print(results_semantic_sim_answer_optimized)

# Salva le liste in un file pickle
with open("pickle_data/results_semantic_sim_expt_cot_llava.pkl", "wb") as f:
    pickle.dump(results_semantic_sim_expt_cot, f)

with open("pickle_data/results_semantic_sim_answer_original_llava.pkl", "wb") as f:
    pickle.dump(results_semantic_sim_answer_original, f)

with open("pickle_data/results_semantic_sim_answer_optimized_llava.pkl", "wb") as f:
    pickle.dump(results_semantic_sim_answer_optimized, f)

#################################################### BLEU Score 
# Funzione asincrona per calcolare il BLEU score
async def calculate_bleu(result_answer):
    # Esempio di dati: una risposta del modello e una risposta di riferimento
    scores=[]
    index=0

    for key, value in dataset_reference.items():
        if "inductive" in value['skill'] or "numerical" in value['skill']:
            sample = SingleTurnSample(
            response=preprocess_text(result_answer[index]), 
            reference=preprocess_text(value['reasoning'])
            )
        
            # Creazione dell'oggetto BleuScore
            scorer = BleuScore()

            # Calcolo del BLEU score per il turno singolo (deve essere await)
            score = await scorer.single_turn_ascore(sample)
            scores.append(score)
            index=index+1

        #Visualizzare il risultato
        #print(f"BLEU Score: {score}")

    return scores
# Esegui la funzione asincrona
results_bleu_expt_cot=asyncio.run(calculate_bleu(results_answer_Exp_CoT))
results_bleu_answer_original=asyncio.run(calculate_bleu(results_answer_original))
results_bleu_answer_optimized=asyncio.run(calculate_bleu(results_answer_optimized))

# Salva le liste in un file pickle
with open("pickle_data/results_bleu_expt_cot_llava.pkl", "wb") as f:
    pickle.dump(results_bleu_expt_cot, f)

with open("pickle_data/results_bleu_answer_original_llava.pkl", "wb") as f:
    pickle.dump(results_bleu_answer_original, f)

with open("pickle_data/results_bleu_answer_optimized_llava.pkl", "wb") as f:
    pickle.dump(results_bleu_answer_optimized, f)

print("\n\nBLEU_SCORES risposte ottimizzate con Expert Prompting e Chain of Thought:")
print(results_bleu_expt_cot)
print("BLEU_SCORES risposte originali senza ottimizzazione:")
print(results_bleu_answer_original)
print("BLEU_SCORES risposte ottimizzate (Expert Prompting,ToT,CoT):")
print(results_bleu_answer_optimized)

#################################################### ROUGE Score 
async def calculate_rouge(result_answer):
    # Esempio di dati: una risposta del modello e una risposta di riferimento
    scores=[]
    index=0

    for key, value in dataset_reference.items():
        if "inductive" in value['skill'] or "numerical" in value['skill']:
            sample = SingleTurnSample(
            response=preprocess_text(result_answer[index]), 
            reference=preprocess_text(value['reasoning'])
            )
        
            # Creazione dell'oggetto RougeScore
            scorer = RougeScore()

            # Calcolo del Rouge score per il turno singolo (deve essere await)
            score = await scorer.single_turn_ascore(sample)
            scores.append(score)
            index=index+1

    return scores

# Esegui la funzione asincrona
results_rouge_expt_cot=asyncio.run(calculate_rouge(results_answer_Exp_CoT))
results_rouge_answer_original=asyncio.run(calculate_rouge(results_answer_original))
results_rouge_answer_optimized=asyncio.run(calculate_rouge(results_answer_optimized))

# Salva le liste in un file pickle
with open("pickle_data/results_rouge_expt_cot_llava.pkl", "wb") as f:
    pickle.dump(results_rouge_expt_cot, f)

with open("pickle_data/results_rouge_answer_original_llava.pkl", "wb") as f:
    pickle.dump(results_rouge_answer_original, f)

with open("pickle_data/results_rouge_answer_optimized_llava.pkl", "wb") as f:
    pickle.dump(results_rouge_answer_optimized, f)

print("\n\nROUGE_SCORES risposte ottimizzate con Expert Prompting e Chain of Thought:")
print(results_rouge_expt_cot)
print("ROUGE_SCORES risposte originali senza ottimizzazione:")
print(results_rouge_answer_original)
print("ROUGE_SCORES risposte ottimizzate (Expert Prompting,ToT,CoT):")
print(results_rouge_answer_optimized)
