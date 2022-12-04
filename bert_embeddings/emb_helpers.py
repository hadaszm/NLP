import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForPreTraining, BertModel
import torch
import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")


def return_embeddings(my_sentence, model): 
    input_ids = tokenizer(my_sentence, return_tensors="pt")
    try:
        model = model.bert
    except AttributeError:
        pass
    output = model(**input_ids)

    try: 
        final_layer = output.last_hidden_state
    except AttributeError: 
        final_layer = output[0]
    return final_layer[0][1:-1].detach().numpy().mean(0).reshape(1, -1)
    
def process_extracted_keywords_file(filename): 
    df = pd.read_csv(filename)\
        .drop('Unnamed: 0', axis=1)
    df['topic_keywords'] = df['topic_keywords'].apply(eval)
    df = df.explode('topic_keywords')
    df[['topic_keyword', 'probability']] = df[['topic_keywords']].apply(lambda x: [x['topic_keywords'][0], x['topic_keywords'][1]], result_type='expand', axis=1)
   
    df = df.drop('topic_keywords', axis=1)
    return df

def return_embeddings_for_keywords(keywords, model):
    unique_keywords = keywords.drop_duplicates()
    embeddings = []
    for keyword in tqdm.tqdm(unique_keywords['topic_keyword']): 
        embeddings.append(return_embeddings(keyword, model))
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = pd.DataFrame(embeddings, columns=list(range(embeddings.shape[1])))
    embeddings['topic_keyword'] = unique_keywords['topic_keyword'].reset_index(drop=True)

    return embeddings

def add_embeddings_to_keywords(df, model): 
    keywords_embeddings = return_embeddings_for_keywords(df[['topic_keyword']], model)
    return df.merge(keywords_embeddings)


def return_embeddings_for_concepts(concepts, model):
    embs = []

    for concept in tqdm.tqdm(concepts.values.flatten()): 
        embs.append(return_embeddings(concept, model))

    return pd.DataFrame(np.concatenate(embs), columns=list(range(embs[0].shape[1])))



