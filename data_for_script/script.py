import sys
import os
import nltk
from gensim import corpora, models
import os
import pandas as pd
import copy
from tqdm import tqdm
import math
import numpy as np



def get_now_str():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def transform_strings_to_arrays(df, col_names):
    for col in col_names:
        df[col] = df[col].apply(eval)
    return df
    

def parse_args():
    """
    Function that returns parsed arguments from console.
    """
    arguments = {}
    for arg_ in sys.argv[1:]:
        try:
            key, value = arg_.split('=')
            key = key.strip()
            value = value.strip()
            if key in {'prepare_data'}:
                arguments[key] = eval(value)
            else:
                arguments[key] = value
        except ValueError as e:
            raise e
                
    return arguments

# python script.py prepare_data=False timestamp=2023-01-12_19-45-43
def main():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    data_folder = 'data'
    data_preprocessing_results = 'data/data_preprocessing_results'
    data_path = "data_processed_whole.csv" 
    models_path = "models"
    results_path = "results"
    embeddings_path = "results/embeddings"
    dataset = 'MedM'
    CRAFT_ONTOLOGIES = ['CHEBI', 'CL', 'GO', 'MONDO', 'MOP', 'NCBITAXON', 'PR', 'SO', 'UBERON']
    UMLS_NCBO_ST21pv_ontologies_ids = ['CPT', 'FMA', 'GO', 'HGNC', 'HP', 'ICD10', 'ICD10CM', 'ICD9CM', 'MEDDRA', 'MESH', 'NCBITAXON', 'NCIT', 'NDDF', 'NDFRT', 'OMIM', 'RXNORM', 'SNMI']
    ground_truth_path = 'data/concepts_per_document.csv'
    args = parse_args()
    timestamp = args.get('timestamp', get_now_str())
    
    emb_glob_regex = 'results/embeddings/concepts_embeddings_*_biobert.csv'
    # prepare data 
    if args['prepare_data']: 
        prepared_data_path = prepare_data(data_folder,data_preprocessing_results,dataset)
    else: 
        prepared_data_path = os.path.join(data_preprocessing_results, data_path)
        print(f"Prepared data loaded from path: {prepared_data_path}")
    
    # extract keywords bertopic
    print("Extracting keywords using BERTopic...")
    extracted_keywords_path_bertopic = os.path.join(results_path, 'bertopic', f'bertopic_keywords_{timestamp}.csv')
    if not os.path.exists(extracted_keywords_path_bertopic):
        extracted_keywords_path_bertopic, model_path_bertopic = get_keywords_bertopic(prepared_data_path, models_path, results_path, timestamp)
        print(f"BERTopic model saved to {model_path_bertopic}")
        print(f"BERTopic extracted keywords saved to {extracted_keywords_path_bertopic}")
    else: 
        print(f"Loaded extracted keywords from {extracted_keywords_path_bertopic}")

    # extract keywords lda
    print("Extracting keywords using LDA...")
    extracted_keywords_path_lda = os.path.join(results_path, 'lda',  f'LDA_{timestamp}.csv')
    if not os.path.exists(extracted_keywords_path_lda):
        extracted_keywords_path_lda, model_path_lda = get_keywords_lda(prepared_data_path, models_path, results_path, timestamp)
        print(f"LDA model saved to {model_path_lda}")
        print(f"LDA extracted keywords saved to {extracted_keywords_path_lda}")
    else: 
        print(f"Loaded extracted keywords from {extracted_keywords_path_lda}")

    print('\nExtracting keywords done. Starting tagging.')
    
        
    print("Embedding tagger")
    print("Calculating embeddings of keywords...")
    keyword_files_input = {
        'bertopic': extracted_keywords_path_bertopic,
        'LDA': extracted_keywords_path_lda
    }
      
    keyword_files_output = {
        name: file.replace(name.lower() + '\\', 'embeddings\\') for name, file in keyword_files_input.items()
    }
    
    if all([not os.path.exists(filename) for name, filename in keyword_files_output.items()]):
        print("Calculating embeddings for ncbo")
        keyword_files_output = calculate_embeddings_for_keywords(list(keyword_files_input.values()))
        for name, filename in keyword_files_output.items():
            print(f"Embeddings for {name} are saved in {filename}")
    else: 
        for name, filename in keyword_files_output.items():
            print(f"Embeddings for {name} are loaded from {filename}")
    

    
    emb_files_input = keyword_files_output
    
    emb_files_output = {}
    for name, filename in emb_files_input.items():
        output_filename = filename.replace('embeddings', 'emb_tagger')
        output_filename = os.path.join(os.path.dirname(output_filename), f'tagged_{timestamp}_' + os.path.basename(output_filename))
        emb_files_output[name] = output_filename
    print('emb_files_output', emb_files_output)

    print('{filename: os.path.exists(filename) for filename in emb_files_output.values()}', {filename: os.path.exists(filename) for filename in emb_files_output.values()})
    if all([not os.path.exists(filename) for filename in emb_files_output.values()]):
        print("Embedding tagging")
        emb_files_output = emb_tagger(emb_files_input, emb_glob_regex, timestamp)
        for name, filename in emb_files_output.items():
            print(f"Output for {name} is saved in {filename}")
    else: 
        for name, filename in emb_files_output.items():
            print(f"Output for {name} is loaded from {filename}")
    
    print("NCBO tagger for BERTopic")
    ncbo_bertopic_tagged_keywords_path = f'{results_path}/bertopic_ncbo/bertopic_ncbo_{timestamp}.csv'
    if not os.path.exists(ncbo_bertopic_tagged_keywords_path):
        ncbo_bertopic_tagged_keywords_path = tag_ncbo(UMLS_NCBO_ST21pv_ontologies_ids, 
                        'bertopic', 
                        extracted_keywords_path_bertopic, 
                        results_path, 
                        timestamp)
        print(f"NCBO tagging for saved in {ncbo_bertopic_tagged_keywords_path}")
    else: 
        print(f"NCBO tagging for loaded from {ncbo_bertopic_tagged_keywords_path}")
        
    print("NCBO tagger for LDA")
    
    ncbo_lda_tagged_keywords_path = f'{results_path}/lda_ncbo/lda_ncbo_{timestamp}.csv'
    if not os.path.exists(ncbo_lda_tagged_keywords_path):
        ncbo_lda_tagged_keywords_path = tag_ncbo(UMLS_NCBO_ST21pv_ontologies_ids, 
                        'lda', 
                        extracted_keywords_path_lda, 
                        results_path, 
                        timestamp)
        print(f"NCBO tagging for LDA saved in {ncbo_lda_tagged_keywords_path}")
    else: 
        print(f"NCBO tagging for LDA loaded from {ncbo_lda_tagged_keywords_path}")
    
    
    print("Calculating embeddings for ncbo tags")
    ncbo_emb_path = os.path.join(embeddings_path, f'{timestamp}_ncbo_embeddings.csv')
    if not os.path.exists(ncbo_emb_path):
        ncbo_emb_path = calculate_embeddings_from_ncbo([ncbo_lda_tagged_keywords_path,
                                        ncbo_bertopic_tagged_keywords_path], 
                                        timestamp, embeddings_path)
        print(f"NCBO tags' embeddings saved in {ncbo_emb_path}")
    else: 
        print(f"NCBO tags' embeddings were already saved in {ncbo_emb_path}")
    
    print("Disambiguation")
    bertopic_dismbiguation_results_path = os.path.join(results_path,f'disambiguation_res_bertopic_ncbo_{timestamp}.csv')
    if not os.path.exists(bertopic_dismbiguation_results_path):
        bertopic_dismbiguation_results_path = prepare_disambiguation(results_path,
                        ncbo_bertopic_tagged_keywords_path,
                        extracted_keywords_path_bertopic, 
                        ncbo_emb_path,
                        timestamp, 'bertopic_ncbo')
        print(f"BERTopic disambiguation saved in {bertopic_dismbiguation_results_path}")
    else: 
        print(f"BERTopic disambiguation loaded from {bertopic_dismbiguation_results_path}")
    
    lda_dismbiguation_results_path = os.path.join(results_path,f'disambiguation_res_lda_{timestamp}.csv')
    if not os.path.exists(lda_dismbiguation_results_path):
        lda_dismbiguation_results_path = prepare_disambiguation(results_path,
                        ncbo_lda_tagged_keywords_path,
                        extracted_keywords_path_lda, 
                        ncbo_emb_path,
                        timestamp, 'lda_ncbo')
        print(f"LDA disambiguation saved in {lda_dismbiguation_results_path}")
    else: 
        print(f"LDA disambiguation loaded from {lda_dismbiguation_results_path}")
    
    # lda_dismbiguation_results_path
    # bertopic_dismbiguation_results_path
    # emb_files_output
    print(emb_files_output.values())
    print("Saving summary")
    sum_path = summary(ground_truth_path, 
            [lda_dismbiguation_results_path, bertopic_dismbiguation_results_path], 
            list(emb_files_output.values()), 
            emb_glob_regex, timestamp)
    print(f"Saved summary to {sum_path}")

def emb_tagger(emb_files, emb_glob_regex, timestamp):
    import pandas as pd
    import numpy as np
    from numba import jit, njit, prange
    import glob
    import os
    import tqdm
    emb_cols = [str(col) for col in range(767+1)]
    @njit
    def cosine_sim(a, b): 
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    @njit(parallel=True)
    def return_similarities(embedding, concept_embeddings):
        result = np.zeros((concept_embeddings.shape[0], ))
        for i in prange(concept_embeddings.shape[0]):
            result[i] = cosine_sim(embedding, concept_embeddings[i])
        return result
    concepts_emb_pretrained = pd.concat([pd.read_csv(file) for file in glob.glob(emb_glob_regex)]).reset_index(drop=True)
    concepts_emb_pretrained_raw = np.ascontiguousarray(concepts_emb_pretrained[emb_cols].values.astype(np.float32))
    concepts_emb_pretrained = concepts_emb_pretrained[['concept_name', 'CUI']]
    n_words = 1
    def return_embedded_words(row):
        embedding = row[0].astype(np.float32).flatten()
        #cosine_similarities = np.apply_along_axis(lambda x: cosine_sim(embedding, x), axis=1, arr=concepts_emb_pretrained_raw)
        cosine_similarities = return_similarities(embedding, concepts_emb_pretrained_raw)
        argsort_indices = np.argsort(cosine_similarities)
        chosen_concepts_names = set()
        i = 0
        while len(chosen_concepts_names) != n_words: 
            max_indices = argsort_indices[-(n_words+i):]
            chosen_concepts_names = set(concepts_emb_pretrained.iloc[max_indices]['concept_name'])
            i += 1

        return concepts_emb_pretrained.iloc[max_indices]['concept_name'].drop_duplicates(), concepts_emb_pretrained.iloc[max_indices]['CUI']
    def collect_list(x): 
        result= {}
        result['tagged_words'] = list(x['tagged_word'])
        return pd.Series(result, index=['tagged_words'])
    def postprocess_keywords(x):
        result = {}
        result['text_to_annotate'] = ', '.join(x['topic_keyword'])
        result['ncbo_annotations_pairs'] = []
        for topic_keyword, tagged_words in zip(x['topic_keyword'], x['tagged_word']): 
            for tagged_word in tagged_words: 
                result['ncbo_annotations_pairs'].append([topic_keyword, tagged_word])
        return pd.Series(result, index=['text_to_annotate', 'ncbo_annotations_pairs'])
    chosen_concepts_names = set()
    output_filenames = {}
    for name, filename in emb_files.items():
        keyword_embs = pd.read_csv(filename)
        #print(keyword_embs.columns) # PMID,topic_number,topic_probs,topic_keyword,probability
        distict_keywords = keyword_embs.groupby('topic_keyword').apply(lambda x: x[emb_cols].values[0]).reset_index()
        tagged_words = []
        CUIs = []
        similarities = [] 
        for i, row in tqdm.tqdm(distict_keywords.iterrows(), total=len(distict_keywords)):
            tagged_word, CUI = return_embedded_words(row)
            tagged_words.append(tagged_word.values)
            CUIs.append(CUI.values)
            chosen_concepts_names.update(tagged_word.values)
        distict_keywords['tagged_word'] = tagged_words
        distict_keywords['CUI'] = CUIs
        keyword_embs = keyword_embs.merge(distict_keywords)
        output_filename = filename.replace('embeddings', 'emb_tagger')
        output_filename = os.path.join(os.path.dirname(output_filename), f'tagged_{timestamp}_' + os.path.basename(output_filename))\
            .replace('emb_tagger_keywords', 'bertopic_keywords')

        #keyword_embs.groupby('PMID').apply(collect_list).reset_index().to_csv('../results/tagged_bertopic_bert_pretrained.csv', index=False)

        keyword_embs.drop([str(col) for col in list(range(767+1))] + [0],axis=1)\
            .groupby('PMID').apply(postprocess_keywords).reset_index().to_csv(output_filename, index=False)
        output_filenames[name] = output_filename
    return output_filenames
    
def summary(ground_truth_path, disambiguation_results, emb_results, emb_glob_regex, timestamp): 
    import pandas as pd 
    import numpy as np
    import glob
    import os
    import copy
    def precision(TP, FP): 
        return TP/(TP+FP)

    def recall(TP, FN): 
        return TP/(TP+FN)

    def F1(precision, recall):
        return 2/(1/precision + 1/recall)

    def return_TP_FN_FP(keywords, ground_truth):
        keywords = set(keywords)
        ground_truth = set(ground_truth)
        TP = len(keywords.intersection(ground_truth) )
        FN = len(keywords.difference(ground_truth))
        FP = len(ground_truth.difference(keywords))
        return np.array([TP, FN, FP])
    def explode_list(pmid, pairs):
        result = []
        for pair in pairs:
            result.append((pmid, pair[0], pair[1]))
        return result

    def explode_df(df): 
        result = []
        for i, row in df.iterrows(): 
            result += explode_list(row['PMID'], eval(row['ncbo_annotations_pairs']))
        return pd.DataFrame(result, columns=['PMID', 'keyword', 'tag'])

    gt = pd.read_csv(ground_truth_path)\
        .rename({'Concepts': 'true_CUIs'}, axis=1)\
        .drop('Unique_concepts', axis=1)
    gt['true_CUIs'] = gt['true_CUIs'].apply(eval).apply(set).apply(list).apply(lambda cuis: [cui[5:] for cui in cuis])
    concepts_emb = pd.concat([pd.read_csv(file) for file in glob.glob(emb_glob_regex)]).reset_index(drop=True)
    concepts_to_cuis = concepts_emb.groupby('concept_name').apply(lambda x: list(x['CUI'])).reset_index()
    concepts_to_cuis = concepts_to_cuis.rename({0: 'CUIs'}, axis=1)
    del concepts_emb
    def concat_cuis(x): 
        result = {
            'pred_cuis': sum(x['CUIs'], []),
            'true_cuis': sum(x['true_CUIs'], [])
        }
        return pd.Series(result, index=['pred_cuis', 'true_cuis'])
    #UMLS_ST21pv_semantic_types_ids = {'T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 'T038',
    #'T058', 'T062', 'T074', 'T082', 'T091', 'T092', 'T097', 'T098', 'T103', 'T168', 'T170', 'T201', 'T204'}
    #s_types = pd.read_csv(cui_type_path, sep='|', header=None, dtype=str)
    #s_types = s_types[[0, 1]].rename({0: 'CUI', 1: 'type'}, axis=1)
    #s_types = s_types.loc[s_types['type'].isin(UMLS_ST21pv_semantic_types_ids)]
    #s_types = s_types.groupby('CUI').apply(lambda x: list(x['type'])).reset_index()\
    #    .rename({0: 'types'}, axis=1)
    #
    #s_types = dict(zip(s_types['CUI'], s_types['types']))
    #def change_cui_to_type(x):
    #    result = {}
    #    result['pred_type'] = [s_types[cui] for cui in set(x['pred_cuis']) if cui in s_types]
    #    result['true_type'] = [s_types[cui] for cui in set(x['true_cuis']) if cui in s_types]
    #    
    #    result['pred_type'] = sum(result['pred_type'], [])
    #    result['true_type'] = sum(result['true_type'], [])
    #
    #    return pd.Series(result, index=['pred_type', 'true_type'])
    stats = []
    for file in emb_results:
        tagged = explode_df(pd.read_csv(file))
        tagged = tagged.merge(concepts_to_cuis, left_on='tag', right_on='concept_name', how='left')\
            .merge(gt, how='left')

        tagged = tagged.groupby('PMID').apply(concat_cuis).reset_index()

        #tagged[['pred_type', 'true_type']] = tagged[['pred_cuis','true_cuis']].apply(change_cui_to_type, axis=1)

        TP_FN_FP_cuis = np.zeros((3,))
        #TP_FN_FP_types = np.zeros((3,))
        for i, row in tagged.iterrows(): 
            TP_FN_FP_cuis += return_TP_FN_FP(row['pred_cuis'], row['true_cuis'])
            #TP_FN_FP_types += return_TP_FN_FP(row['pred_type'], row['true_type'])

        precision_ = precision(TP_FN_FP_cuis[0], TP_FN_FP_cuis[2])
        recall_ = recall(TP_FN_FP_cuis[0], TP_FN_FP_cuis[1])
        stats.append(('cui', os.path.basename(file), precision_, recall_, F1(precision_, recall_)))

        #precision_ = precision(TP_FN_FP_types[0], TP_FN_FP_types[2])
        #recall_ = recall(TP_FN_FP_types[0], TP_FN_FP_types[1])
        #stats.append(('type', os.path.basename(file), precision_, recall_, F1(precision_, recall_)))

    #pd.DataFrame(stats, columns=['agg_level', 'file', 'precision', 'recall', 'F1']).sort_values(['agg_level', 'file'])
    print(disambiguation_results)
    for file in disambiguation_results:
        df = pd.read_csv(file)
        df = df[['PMID', 'disambiguation_best_concept']]
        df['disambiguation_best_concept'] = df['disambiguation_best_concept']\
            .apply(eval).apply(lambda x: list(x.values()))
        df = df.explode('disambiguation_best_concept')
        df['disambiguation_best_concept'] = df['disambiguation_best_concept'].apply(lambda x: x[0])
        
        df = df.merge(concepts_to_cuis, left_on='disambiguation_best_concept', right_on='concept_name', how='left')\
            .drop('disambiguation_best_concept', axis=1)\
            .merge(gt, how='left')\
            .rename({'CUIs': 'pred_cuis', 'true_CUIs': 'true_cuis'}, axis=1)
        df['pred_cuis'] = df['pred_cuis'].apply(lambda x: [] if x != x else x)
        #df[['pred_type', 'true_type']] = df[['pred_cuis','true_cuis']].apply(change_cui_to_type, axis=1)

        TP_FN_FP_cuis = np.zeros((3,))
        #TP_FN_FP_types = np.zeros((3,))
        for i, row in df.iterrows(): 
            TP_FN_FP_cuis += return_TP_FN_FP(row['pred_cuis'], row['true_cuis'])
            #TP_FN_FP_types += return_TP_FN_FP(row['pred_type'], row['true_type'])

        precision_ = precision(TP_FN_FP_cuis[0], TP_FN_FP_cuis[2])
        recall_ = recall(TP_FN_FP_cuis[0], TP_FN_FP_cuis[1])
        stats.append(('cui', os.path.basename(file), precision_, recall_, F1(precision_, recall_)))

        #precision_ = precision(TP_FN_FP_types[0], TP_FN_FP_types[2])
        #recall_ = recall(TP_FN_FP_types[0], TP_FN_FP_types[1])
        #stats.append(('type', os.path.basename(file), precision_, recall_, F1(precision_, recall_)))
    stats = pd.DataFrame(stats, columns=['agg_level', 'file', 'precision', 'recall', 'F1'])
    output = f'results/results_{timestamp}.csv'
    stats.sort_values(['agg_level', 'file']).to_csv(output, index=False)
    return output


def calculate_embeddings_for_keywords(files):
    from emb_helpers import return_embeddings_for_concepts 
    from transformers import AutoTokenizer, AutoModel
    import torch
    import glob
    import pandas as pd
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    biobert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device_name)
    
    df = pd.concat([pd.read_csv(file) for file in files])
    all_keywords = sum(df['topic_keywords'].apply(eval), [])
    all_keywords = list(set([keyword for keyword, prob in all_keywords]))
    all_keywords = pd.DataFrame(all_keywords, columns=['keyword'])
    embeddings = return_embeddings_for_concepts(all_keywords['keyword'],biobert, 'cpu')
    embeddings['keyword'] = all_keywords['keyword']
    embeddings = embeddings[['keyword'] + list(range(len(embeddings.columns)-1))]
    output_files = {}
    for file in files: 
        method = 'bertopic' if 'bertopic' in file else 'lda'
        df = pd.read_csv(file)
        df['topic_keywords'] = df['topic_keywords'].apply(eval)
        df = df.explode('topic_keywords')
        df['topic_keyword'] = df['topic_keywords'].apply(lambda x: x[0])
        df['probability'] = df['topic_keywords'].apply(lambda x: x[1])
        df = df.drop('topic_keywords', axis=1)
        df = df.merge(embeddings, left_on='topic_keyword', right_on='keyword')
        output_file = file.replace(method, 'embeddings')
        df.to_csv(output_file, index=False)
        output_files[method] = output_file
    return output_files

def calculate_embeddings_from_ncbo(files, timestamp, embeddings_output_filepath): 
    from emb_helpers import return_embeddings_for_concepts 
    from transformers import AutoTokenizer, AutoModel
    import torch
    import glob
    import pandas as pd
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    biobert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device_name)
    
        
    df = pd.concat([pd.read_csv(file) for file in files])
    df['ncbo_annotations_pairs'] = df['ncbo_annotations_pairs'].apply(lambda x: sum([row[:2] for row in eval(x)], []))
    all_words_to_embed = list(set(sum(df['ncbo_annotations_pairs'], [])))
    all_words_to_embed = pd.DataFrame(all_words_to_embed, columns=['words'])
    embeddings = return_embeddings_for_concepts(all_words_to_embed['words'],biobert, device_name)
    embeddings['words'] = all_words_to_embed['words']
    embeddings[['words'] + list(range(len(embeddings.columns)-1))]

    basename = f'{timestamp}_ncbo_embeddings.csv'
    output_path = os.path.join(embeddings_output_filepath, basename)
    embeddings.to_csv(output_path, index=False)
    return output_path

#-------------------------DISAMBIGUATION-------------------------------------

def get_embeding(word,emb):
    """Get word embedding """
    return emb.loc[word]
def create_tags_list_dict(row, with_sorting = False):
    '''
    Parameters
    ----------
    row : list
        list of lists keyword, concept
    with_sorting : bool
        should sorting be enables
        
    Returns
    ------
    results : dict
        initaly sorted (or random ordered) dictionary keyword : dict of concepts {concept : distance}
    '''
    result = {}
    for tag in row:
        key = tag[0].split(',')[0].upper()
        value = tag[1].upper()
        if key in result.keys():
            if  value not in result[key]:
                result[key][value]=1
            else:
                 result[key][value]+=1
        else:
            result[key]= {value:1}
    if with_sorting:
        for key in result.keys():
            d = result[key]
            d = {k:0 for k,v in dict(sorted(d.items(), key=lambda item: item[1], reverse = True)).items()}
            result[key] = d
    else:
        for key in result.keys():
            d = result[key]
            d = {k:0 for k,v in d.items()}
            result[key] = d
        

    return result
def disambiguation(current_selection,embedings, weigths, forced):
    ''' 
    Parameters
    ----------
    current_selection : dict
         dictionary keyword: list of all unique concepts
    weigths: dict
         the importance of given keyword
    forced: bool
        should the concept identical with keyword be the first

    Returns
    ------
    new_current_selction : dict
        dictionary with new best selction of concepts
    
    '''
    # we iterate over the current_selection MAX_ITER times
    new_current_selction  = copy.deepcopy(current_selection)
    should_stop = False
    for i in range(7):

        for keyword, concepts_list in new_current_selction.items():
            distances = {} # for each possible concept calaculate the mean distance from other kewords (concepts of them)
            if forced and  any([c== keyword for c in concepts_list.keys()]):
                _ = [c != keyword for c in concepts_list.keys()]
                distances = dict(zip(_,[1000]* len(_)))
                distances[keyword] = 0
            else:
                for concept in concepts_list.keys():
                    distances[concept] = []
                    for k, current_best_tags in new_current_selction.items():
                        # foreach keyword that is not a current one 
                        if k!=keyword:
                            current_best_tag = list(current_best_tags.keys())[0] # the first out of list of concepts
                            try:
                                distances[concept].append(weigths[k]*math.dist(get_embeding(concept,embedings),get_embeding(current_best_tag,embedings))) # append distance from this concept
                            except Exception as e:
                                print(e)
                    distances[concept] = np.mean(distances[concept]) # mean distance 
            new_current_selction[keyword] = dict(sorted(distances.items(), key=lambda item: item[1]))  # upadate the current selection of this keyword
    return new_current_selction
    
    
def keywords_importance(grouped_data, tagger_data):
    """get keywords importance"""
    return grouped_data.reset_index().merge(tagger_data[['PMID','topic_keywords']] ,on = 'PMID').set_index('text_to_annotate')

def get_n_best_tags(data, n = 1):
    """how many best concepts to take"""
    return [{k:sorted(v, key=v.get)[:n] for k,v in dd.items()} for dd in data['after_disambiguation']]

def prepare_for_disambiguation(data, tagger, embedings,column_name = 'ncbo_annotations_pairs' ,  weighting = False, sorting = False, forced = False, take_best = 1):
    """prepare data for disambiguation - read csv, eval, set index etc.

    Parameters
    ----------
    datas : DataFrames
        DataFrame with keywords
    tagger : DataFrames
        DataFrame with keywords importance
    embedings : DataFrames
        DataFrame with keywords embedings
    column_name : str
        Name of the column with annotations
    weigthing: bool
        Should the weigthed voting be performed
    sorting: bool
        Should the initial sorting be performed
    take_best : int
        How many best concepts should be returned



    Returns
    ------
    data  : DataFrame
    """
    grouped = data.groupby('text_to_annotate').nth(0)
    # get importance for each keyword -> will be used if weighting True
    grouped = keywords_importance(grouped, tagger )
    grouped['possible_tags'] = grouped[column_name].apply(lambda r: create_tags_list_dict(r, sorting))

    # disambiguation
    res = []
    for idx, row  in tqdm.tqdm(grouped.iterrows(), total = len(grouped)):
        current_selection = row['possible_tags']
        if not weighting:
            weigths = dict(zip(list(row['topic_keywords'].keys()),[1] * len(row['topic_keywords'])))
        else:
            weigths = row['topic_keywords']
        r = disambiguation(current_selection, embedings,weigths,forced)
        res.append(r)
    grouped['after_disambiguation'] = res
    data = data.merge(grouped['after_disambiguation'].reset_index(), on = 'text_to_annotate' )
    data['disambiguation_best_concept'] = get_n_best_tags(data, take_best)
    return data


def prepare_data(data_name, tagger_name, embedings_name):
    """prepare data for disambiguation - read csv, eval, set index etc.

    Parameters
    ----------
    data_name : str
        Path to get data
    tagger_name : str
        Path to save the results to (folder must exist).

    embedings_name : str
        Path to save the results to (folder must exist).


    Returns
    ------
    data, tagger, embedings : tuple (DataFrame, DataFrame, DataFrame) 
    """
    import pandas as pd
    data = pd.read_csv(data_name)
    data['ncbo_annotations_pairs'] = data['ncbo_annotations_pairs'].apply(eval)
    data['ncbo_annotations_pairs']  = data['ncbo_annotations_pairs'].apply(lambda x : [[a[0].upper(),a[1]] for a in x])

    tagger = pd.read_csv(tagger_name)
    tagger['topic_keywords'] = tagger['topic_keywords'].apply(eval).apply(lambda x: {k.upper():v for k,v in dict(x).items()})


    embedings = pd.read_csv(embedings_name)
    embedings = embedings.set_index('words')
    embedings.index = embedings.index.str.upper()
    embedings = embedings[~embedings.index.duplicated(keep='first')]

    return  data, tagger, embedings


def prepare_disambiguation(results_folder,data_path, tagger_path, embedings_path,timestamp,weigthing=False,sorting=False,forced = False):
    """Performs disambiguation

    Parameters
    ----------
    results_folder : str
        Path to save results
    
    data_path : str
        Path to save the model to (folder must exist).

    tagger_path : str
        Path to save the results to (folder must exist).

    embedings_path : str
        Path to save the results to (folder must exist).

    timestamp : str
        Timestamp that will be added to filenames

    weigthing: bool
        Should the weigthed voting be performed

    sorting: bool
        Should the initial sorting be performed

    forced: bool
        If true: if the any concept is identical with keyword it is returned


    Returns
    ------
    result_path : str
        Path to results
    """

    data, tagger, embedings = prepare_data( data_path,tagger_path,embedings_path)
    data = prepare_for_disambiguation(data,tagger,embedings,'ncbo_annotations_pairs',weigthing,sorting,forced)
    data.to_csv(results_folder)

    return results_folder

#-------------------------LDA------------------------------------------------


def get_lda_results(data_path, num_topics = 10,num_keywords = 10):
    """Performs lda keywords extraction for data after lemmatization.

    Parameters
    ----------
    data_path : str
        Path to preprocessed dataset. Dataset must contain a column with name 'tokenized_words_lemmatize'.
    timestamp : str
        Timestamp of getting data
    num_topic : int
        Number of disired topics

    num_keywords : int
        Number of keywords per topic

    Returns
    ------
    result, topic_distribution,lda_model : (DataFrame,DataFrame,model)
        DataFrame with reults, DataFrae with topic distribution and lda_model

    """

    def get_topic_distribution(lda_model, number_of_topics, number_of_keywords):
        topics_distrib = {}
        for t in lda_model.print_topics(number_of_topics,number_of_keywords):
            topics_distrib[t[0]] =[(a.split('*')[1][1:-1],float(a.split("*")[0])) for a in t[1].split(' + ')]
        return topics_distrib


    data = pd.read_csv(data_path)
    columns = ['tokenized_sentences', 'tokenized_words_lemmatize']
    for col in columns:
        data[col] = data[col].apply(eval)

    texts = data.groupby('PMID')['tokenized_words_lemmatize'].agg(lambda x: x.iloc[0]+x.iloc[1])
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    
    lda_model = models.LdaMulticore(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=num_topics,
                                        passes = 20)
    doc_lda = lda_model[corpus]

    topic_distribution = get_topic_distribution(lda_model,num_topics,num_keywords)
    topics_results = pd.DataFrame.from_records([topic_distribution]).T.reset_index().rename(columns = {'index':'topic_number',0:'topic_keywords'})
    topics_results['keywords'] = topics_results['topic_keywords'].apply(lambda x: [a[0] for a in x])

    

    docs= []
    for doc in doc_lda:
        docs.append({
            'topic_number':doc[0][0],
            'topic_probs': float(doc[0][1]),
            'topic_keywords': topics_results.iloc[doc[0][0]]['topic_keywords'],
            'keywords': topics_results.iloc[doc[0][0]]['keywords']

        })

    docs = pd.DataFrame.from_records(docs)

    results = data[['PMID']].drop_duplicates().reset_index(drop=True).join(docs)
    topics_results = pd.DataFrame.from_records([topic_distribution]).T.reset_index().rename(columns = {'index':'topic_number',0:'topic_keywords'})



    return results,topics_results, lda_model

def get_keywords_lda(data_path, models_path, results_path, timestamp, num_topics = 10,num_keywords = 10):
    """Performs lda keywords extraction for data after lemmatization.

    Parameters
    ----------
    data_path : str
        Path to preprocessed dataset. Dataset must contain a column with name 'tokenized_words_lemmatize'.
    
    models_path : str
        Path to save the model to (folder must exist).

    results_path : str
        Path to save the results to (folder must exist).

    timestamp : str
        Timestamp that will be added to filenames

    num_topic : int
        Number of disired topics

    Returns
    ------
    (result_path, model_save_name) : tuple
        Frist element is the path to created file with extracted keywrods, second - path to created model.
    """

    results,topics_results, lda_model =  get_lda_results(data_path, num_topics ,num_keywords)

    results_path = os.path.join(os.path.join(results_path, f'lda_results_{timestamp}.csv'))
    results.to_csv(results_path)

    models_path = os.path.join(models_path,f"lda_model_{timestamp}")
    lda_model.save(models_path)

    return results_path,models_path
    
#-----------------------DATA PREPERATION-------------------------------
def prepare_data(data_folder,results_folder,option):
    """Performs bertopic keywords extraction for data after lemmatization.

    Parameters
    ----------
    data_folder : str
        Path to folder with files with data (filenames are hardcoded)

    results_folder : str
        Path to folder where results will be saved (folder must exists)

    option : str
        MedM if MedMentions dataset, CRAFT if craft

    Returns
    ------
    result_path : str
        Path to results
    """
    import gzip
    import pandas as pd
    import tqdm
    import re
    import os
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import xml.etree.ElementTree as ET
    from tqdm import tqdm
  

    def parse(file_path:str, data_columns:list,annotations_columns:list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        data = pd.DataFrame(columns = data_columns)
        annotations = pd.DataFrame(columns = annotations_columns)
        errors = pd.DataFrame()
        # hardcoded -  differentiate if the line contains content or annotation
        # if HEADED it is content
        HEADER = re.compile(r"(?P<PMID>[0-9]*)\|(?P<Type>[t|a])\|(?P<Content>.*)")
        with gzip.open(file_path, 'rb') as f:
            i = 0
            for line in tqdm(f.readlines()): 
                i+=1
                l = line.decode("utf-8")
                if l == '\n':
                    continue
                h = HEADER.match(l)
                if h:
                    data = pd.concat([data,pd.DataFrame([{k:h.group(k) for k in data_columns}])], ignore_index=True)
                else:
                    _ = l.split('\t')
                    if len(_) == len(annotations_columns):
                        annotations = pd.concat([annotations,pd.DataFrame([dict(zip(annotations_columns,_))])], ignore_index=True)
                    else:
                        errors = pd.concat([errors,pd.DataFrame([l])],ignore_index=True)
        return data, annotations,errors


    def process_lemma(lemmatizer,sentence: list) -> list:
        """
        takes list of tokens and returns steamed tokens without stopwords
        If word contains non-letters it appends it to the final list
        """
        processed = []
        for word in sentence:
            try:
                word_lower = word.lower() 
                if word_lower not in stopwords.words():
                    processed.append(lemmatizer.lemmatize(word))
            except TypeError: # when word contains non-letters
                processed.append(word)
        return processed


    def get_folder_for_ontology(ontology):
        folder = os.path.join('concept-annotation',ontology,ontology,'knowtator')
        return folder


    def get_data_from_file(root):
        annotations = pd.DataFrame(columns=['StartIndex', 'EndIndex','MentionTextSegment','EntityID'])
        for child in root:
        # annotation
            if child.tag=='annotation':
                tmp = {}
                id_name = None
                for c in child:
                    if c.tag == 'mention':
                        id_name = c.attrib['id']
                    elif c.tag == 'span':
                        tmp['StartIndex'] = c.attrib['start']
                        tmp['EndIndex'] = c.attrib['end']
                annotations.loc[id_name,['StartIndex','EndIndex']] = tmp

        # classmention
            else:
                id_name =child.attrib['id']
                tmp = {}
                for c in child:
                    if c.tag == 'mentionClass' and 'id' in c.attrib.keys():
                        tmp['MentionTextSegment'] = c.text
                        tmp['EntityID'] = c.attrib['id']
                        annotations.loc[id_name,['MentionTextSegment','EntityID']] = tmp
        return annotations



    def get_data(texts,ontology,file_name):
        folder = get_folder_for_ontology(ontology)
        folder = os.path.join(data_folder,folder)
        data = pd.DataFrame()
        for file in os.listdir(folder):
            tree = ET.parse(os.path.join(folder,file))
            root = tree.getroot()
            annotations = get_data_from_file(root)
            annotations['PMID'] = file[:8]
            data = pd.concat([data,annotations])
        file_path = os.path.join('data',ontology,file_name)
        isExist = os.path.exists(os.path.join('data',ontology))
        if not isExist:
            os.makedirs(os.path.join('data',ontology))
        data.to_csv(file_path)
        return data


    
    data_columns = ['PMID', 'Type','Content']
    annotations_columns = ['PMID', 'StartIndex','EndIndex','MentionTextSegment','SemanticTypeID','EntityID']

    if option == 'MedM':

        data_21, annotations_21,errors_21 = parse(os.path.join(data_folder,'corpus_pubtator.txt.gz'), data_columns, annotations_columns)
        semantic_mapping = pd.read_csv(os.path.join(data_folder, 'semantic_type_mapping.txt'), sep = '|', header=None)[[1,2]]
        semanitc_mapper = dict(zip(semantic_mapping[1],semantic_mapping[2] ))
        annotations_21['EntityID'] = annotations_21['EntityID'].apply(lambda x : x.replace('\n',''))
        annotations_21['SemanticMeaning'] = annotations_21['SemanticTypeID'].apply(lambda x : semanitc_mapper[x])
        data_21 = data_21.reset_index()
        annotations_21 = annotations_21.reset_index()
        annotations_21.to_csv(os.path.join(results_folder,'annotations.csv'), index=False)


    if option == 'CRAFT':
        articles_folder = os.path.join('data','articles','txt')
        data_21 = pd.DataFrame(columns = ['PMID','Type','Content'])
        for i,file in enumerate(os.listdir(articles_folder)):
            if file[-3:] == 'txt':
                name = file[:-4]
                with open(os.path.join(articles_folder,file),'r',encoding='utf-8') as f:
                    lines = list(f.readlines())
                    data_21.loc[len(data_21)] = {'PMID':name, 'Type':'t','Content':lines[0]}
                    data_21.loc[len(data_21)] = {'PMID':name, 'Type':'a','Content':''.join(lines[1:])}

        for ontology in os.listdir(os.path.join('data','concept-annotation')):
            get_data(data_21,ontology,'annotations.csv')

    lemmatizer = WordNetLemmatizer()

    # sentences
    data_21['tokenized_sentences'] = data_21['Content'].apply(lambda text : nltk.sent_tokenize(text))
    data_21['tokenized_words'] = None
    data_21['tokenized_words_lemmatize'] = None
    for index, row in tqdm(data_21.iterrows(), total = len(data_21)):
        tokens = []
        tokens_lemma = []
        for sentence in row['tokenized_sentences']:
            tok_sen = nltk.word_tokenize(sentence)
            tokens.append(tok_sen)
            tokens_lemma.append(process_lemma(lemmatizer,tok_sen))
        data_21.at[index, 'tokenized_words_lemmatize'] = tokens_lemma


    for index, row in tqdm(data_21.iterrows(), total = len(data_21)):
        tokens = []
        for sentence in row['tokenized_words_lemmatize']:
            sen = []
            for word in sentence:
                if word.isalnum():
                    tokens.append(word)
            
        data_21.at[index,'tokenized_words_lemmatize']  = tokens


    unique_pmids = data_21['PMID'].drop_duplicates()

    results_path = os.path.join(os.path.join(results_folder,'data_processed_whole.csv'))
    data_21.to_csv(results_path, index=False)

    return results_path


def tag_ncbo(ontologies, keywords_extractor_name, extracted_keywords_path, results_path, timestamp):
    """Performs NCBO tagging for keywords extracted with get_keywords_bertopic or get_keywords_lda functions.

    Parameters
    ----------
    ontologies : list[str]
        List of string of ontologies ids that will be queried in tagging process.
    
    keywords_extractor_name : str
        Name of the algorithm used to extract keywrods (for file/folders naming)

    extracted_keywords_path : str
        Path to the file returned by get_keywords_bertopic or get_keywords_lda functions.

    results_path : str
        Path to save the results to (folder must exist).

    timestamp : str
        timestamp that will be added to filenames

    Returns
    ------
    save_name : str
        Path to tagged words file.
    """

    import urllib.request, urllib.error, urllib.parse
    import json
    import os
    from pprint import pprint
    import pandas as pd
    import re
    REST_URL = "http://data.bioontology.org"
    API_KEY = "194c9635-ce67-4e70-81c5-898c3a2b30fb"

    def read_keywords_extraction_results(path):
        data = pd.read_csv(path, index_col=0)
        data = transform_strings_to_arrays(data, col_names = ['topic_keywords'])
        data['text_to_annotate'] = data.topic_keywords.apply(
            lambda row: re.sub(r"[\'\[\]]", "", str([keyword[0] for keyword in row]))
            )
        return data

    def get_json(url):
        opener = urllib.request.build_opener()
        opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
        return json.loads(opener.open(url).read())

    def create_annotation_pairs(sample_row, column_name):
        found_concepts = sample_row[column_name]
        res_ann_pairs= []
        for _, concept in enumerate(found_concepts):
            max_trials = 5
            trials_no = 0
            while trials_no < max_trials:
                try:
                    concept_class = get_json(concept["annotatedClass"]["links"]["self"])
                    concept_class_ancestors = get_json(concept["annotatedClass"]['links']['ancestors'])
                    break
                except:
                    trials_no+=1
                    continue
            if trials_no==max_trials:
                raise Exception("number of unsuccessfull connection attempts is max_trials")
            annotations = concept['annotations']
            # annotations for this class
            for annot in annotations:
                res_ann_pairs.append([annot['text'], concept_class["prefLabel"], 'DIRECT', concept["annotatedClass"]["links"]["self"]])
            # annotations for ancestors
            for annot in annotations:
                for ancestor in concept_class_ancestors:
                    res_ann_pairs.append([annot['text'], ancestor["prefLabel"], 'ANCESTOR', concept["annotatedClass"]['links']['ancestors']])
        unique_ann_pairs = [list(x) for x in set(tuple(x) for x in res_ann_pairs)]
        return unique_ann_pairs

        
    ##########################################################################################################################

    # read data
    data = read_keywords_extraction_results(extracted_keywords_path)

    # annotate data
    data['ncbo_annotations'] \
        = data.text_to_annotate.apply(lambda text:  \
            get_json(REST_URL + f"/annotator?ontologies={','.join(ontologies)}&text=" + urllib.parse.quote(text)))

    data = data.reset_index()[['PMID', 'text_to_annotate', 'ncbo_annotations']]

    data_to_annotate = data[['text_to_annotate', 'ncbo_annotations']]
    data_to_annotate = data_to_annotate.loc[data_to_annotate.astype(str).drop_duplicates().index]
    data_to_annotate['ncbo_annotation_pairs'] = data_to_annotate.apply(create_annotation_pairs, column_name='ncbo_annotations', axis = 1)

    # create annotation pairs
    data_to_annotate[['text_to_annotate', 'ncbo_annotation_pairs']].to_dict()
    text_to_annot_ncbo_pairs = dict(zip(data_to_annotate.text_to_annotate, data_to_annotate.ncbo_annotation_pairs))
    data['ncbo_annotations_pairs'] = data['text_to_annotate'].apply(lambda text: text_to_annot_ncbo_pairs[text])

    # save data
    res_folder = f'{results_path}/{keywords_extractor_name}_ncbo'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    save_name = f'{res_folder}/{keywords_extractor_name}_ncbo_{timestamp}.csv'
    data.to_csv(save_name, index=False)

    return save_name
    
def get_keywords_bertopic(data_path, models_path, results_path, timestamp):
    """Performs bertopic keywords extraction for data after lemmatization.

    Parameters
    ----------
    data_path : str
        Path to preprocessed dataset. Dataset must contain a column with name 'tokenized_words_lemmatize'.
    
    models_path : str
        Path to save the model to (folder must exist).

    results_path : str
        Path to save the results to (folder must exist).

    timestamp : str
        timestamp that will be added to filenames

    Returns
    ------
    (result_path, model_save_name) : tuple[str]
        Frist element is the path to created file with extracted keywrods, second - path to created model.
    """

    import pandas as pd
    from bertopic import BERTopic
    import os 
    import numpy as np

    # basic BertTopic keyword extraction
    def train_transform_save(train_data, model_save_name, min_topic_size=10):
        
        # train transform
        topic_model = BERTopic(min_topic_size=min_topic_size)
        topics, probs = topic_model.fit_transform(train_data.values)

        # save model
        topic_model.save(model_save_name)

        return topic_model, topics, probs


    def load_transform_save(data, model_save_name, results_path):

        # load model
        loaded_model = BERTopic.load(model_save_name)

        # transform for data 
        samples_topics, samples_probs = loaded_model.transform(data.values)
        res_df = pd.DataFrame({
            'PMID': np.unique(data.index),
            'topic_number': samples_topics,
            'topic_probs': samples_probs,
            "topic_keywords": [loaded_model.get_topic(topic_number) for topic_number in samples_topics]
        })
        res_df.to_csv(results_path, index=False)
        return loaded_model, res_df

    ##############################################################################################################################

    full_data = transform_strings_to_arrays(pd.read_csv(data_path), col_names=['tokenized_words_lemmatize'])

    data = full_data.groupby(by = ['PMID'])['tokenized_words_lemmatize'].agg(lambda x: ' '.join(x.values[0] + x.values[1]))

    model_name = f'bertopic_keywords_{timestamp}'
    model_save_name = os.path.join(models_path, model_name)
    result_path = os.path.join(results_path, 'bertopic', f'{model_name}.csv')

    topic_model, topics, probs = train_transform_save(data, model_save_name, min_topic_size=3)
    _, res_df = load_transform_save(data, model_save_name, result_path)

    return result_path, model_save_name 
                              
if __name__ == '__main__':
    main()
    
    
