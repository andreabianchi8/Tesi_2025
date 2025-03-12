import pandas as pd
import asyncio
import re
import ast
from ragas import SingleTurnSample
from ragas.metrics import BleuScore
from ragas.metrics import RougeScore
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain.llms import LlamaCpp
from sentence_transformers import SentenceTransformer
import pickle

dataset_evaluation=pd.read_csv("evaluation_results_algebra_testmini_JANUS.csv")
dataset_reference=pd.read_csv("algebra_testmini.csv")

results_answer_Exp_CoT=[]
results_answer_original=[]
results_answer_optimized=[]

for index, row in dataset_evaluation.iterrows():
    results_answer_Exp_CoT.append(row['answer_optimized_Exp_CoT'])
    results_answer_original.append(row['answer_original'])
    results_answer_optimized.append(row['answer_optimized'])

#################################################### Factual Correctness
# Percorso al modello GGUF
model_path = "models/mistral-7b-instruct-v0.2.Q5_K_S.gguf"

# Carica il modello locale
# evaluator_llm = LlamaCpp(
#     model_path=model_path,
#     temperature=0,  # Risposte più deterministiche
#     max_tokens=512,
#     n_ctx=2048,  # Aumenta se il modello lo supporta
#     verbose=True
# )

async def factual_correctness(result_answer,evaluator_llm):
    scores=[]
    for index, row in dataset_reference.iterrows():
        
        if(index==18):
            break
    # Crea un sample
    sample = SingleTurnSample(
    response=result_answer[index], 
    reference=row['solution']
    )

    # Valuta con RAGAS
    scorer = FactualCorrectness(llm=evaluator_llm)
    score = await scorer.single_turn_score(sample)
    print("Factual Correctness Score:", score)


#asyncio.run(factual_correctness(results_answer_Exp_CoT,evaluator_llm))
#asyncio.run(factual_correctness(results_answer_original,evaluator_llm))
#asyncio.run(factual_correctness(results_answer_optimized,evaluator_llm))




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
    for index, row in dataset_reference.iterrows():
        
        sample = SingleTurnSample(
            response=result_answer[index], 
            reference=row['solution']
        )
        scorer = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(evaluator_embedding))
        score = await scorer.single_turn_ascore(sample)
        scores.append(score)
        #print(f"Semantic similarity score: {score}")

    return scores

# Creazione del wrapper asincrono per SentenceTransformer
evaluator_embedding = AsyncSentenceTransformer('paraphrase-MiniLM-L6-v2')

# Esegui le valutazioni per le risposte
results_semantic_sim_expt_cot = asyncio.run(semantic_similarity(results_answer_Exp_CoT, evaluator_embedding))
results_semantic_sim_answer_original = asyncio.run(semantic_similarity(results_answer_original, evaluator_embedding))
results_semantic_sim_answer_optimized = asyncio.run(semantic_similarity(results_answer_optimized, evaluator_embedding))

# Salva le liste in un file pickle
with open("pickle_data/results_semantic_sim_expt_cot_JANUS.pkl", "wb") as f:
    pickle.dump(results_semantic_sim_expt_cot, f)

with open("pickle_data/results_semantic_sim_answer_original_JANUS.pkl", "wb") as f:
    pickle.dump(results_semantic_sim_answer_original, f)

with open("pickle_data/results_semantic_sim_answer_optimized_JANUS.pkl", "wb") as f:
    pickle.dump(results_semantic_sim_answer_optimized, f)

print("Semantic Similarity risposte ottimizzate con Expert Prompting e Chain of Thought:")
print(results_semantic_sim_expt_cot)
print("Semantic Similarity risposte originali senza ottimizzazione:")
print(results_semantic_sim_answer_original)
print("Semantic Similarity risposte ottimizzate (Expert Prompting,ToT,CoT):")
print(results_semantic_sim_answer_optimized)

#################################################### BLEU Score 
# Funzione asincrona per calcolare il BLEU score
async def calculate_bleu(result_answer):
    # Esempio di dati: una risposta del modello e una risposta di riferimento
    scores=[]
    for index, row in dataset_reference.iterrows():

        sample = SingleTurnSample(
        response=result_answer[index], 
        reference=row['solution']
        )
        
        # Creazione dell'oggetto BleuScore
        scorer = BleuScore()

        # Calcolo del BLEU score per il turno singolo (deve essere await)
        score = await scorer.single_turn_ascore(sample)
        scores.append(score)

        #Visualizzare il risultato
        #print(f"BLEU Score: {score}")

    return scores
# Esegui la funzione asincrona
results_bleu_expt_cot=asyncio.run(calculate_bleu(results_answer_Exp_CoT))
results_bleu_answer_original=asyncio.run(calculate_bleu(results_answer_original))
results_bleu_answer_optimized=asyncio.run(calculate_bleu(results_answer_optimized))

# Salva le liste in un file pickle
with open("pickle_data/results_bleu_expt_cot_JANUS.pkl", "wb") as f:
    pickle.dump(results_bleu_expt_cot, f)

with open("pickle_data/results_bleu_answer_original_JANUS.pkl", "wb") as f:
    pickle.dump(results_bleu_answer_original, f)

with open("pickle_data/results_bleu_answer_optimized_JANUS.pkl", "wb") as f:
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
    for index, row in dataset_reference.iterrows():

        sample = SingleTurnSample(
        response=result_answer[index], 
        reference=row['solution']
        )
        
        # Creazione dell'oggetto RougeScore
        scorer = RougeScore()

        # Calcolo del Rouge score per il turno singolo (deve essere await)
        score = await scorer.single_turn_ascore(sample)
        scores.append(score)


    return scores

# Esegui la funzione asincrona
results_rouge_expt_cot=asyncio.run(calculate_rouge(results_answer_Exp_CoT))
results_rouge_answer_original=asyncio.run(calculate_rouge(results_answer_original))
results_rouge_answer_optimized=asyncio.run(calculate_rouge(results_answer_optimized))

# Salva le liste in un file pickle
with open("pickle_data/results_rouge_expt_cot_JANUS.pkl", "wb") as f:
    pickle.dump(results_rouge_expt_cot, f)

with open("pickle_data/results_rouge_answer_original_JANUS.pkl", "wb") as f:
    pickle.dump(results_rouge_answer_original, f)

with open("pickle_data/results_rouge_answer_optimized_JANUS.pkl", "wb") as f:
    pickle.dump(results_rouge_answer_optimized, f)

print("\n\nROUGE_SCORES risposte ottimizzate con Expert Prompting e Chain of Thought:")
print(results_rouge_expt_cot)
print("ROUGE_SCORES risposte originali senza ottimizzazione:")
print(results_rouge_answer_original)
print("ROUGE_SCORES risposte ottimizzate (Expert Prompting,ToT,CoT):")
print(results_rouge_answer_optimized)
