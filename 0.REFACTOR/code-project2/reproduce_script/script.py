from common import get_now_str, parse_args

import nltk
import os
import glob
from get_keywords_bertopic import get_keywords_bertopic
from lda import get_keywords_lda
from tag_ncbo import tag_ncbo
from disambiguation import prepare_disambiguation
import pandas as pd


def main():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    prepared_data_path = "0.RESULTS/preprocessing/data_whole.csv"
    models_path = "0.RESULTS/models/"
    results_path = "0.RESULTS"
    embeddings_path = "0.RESULTS/embeddings"
    emb_tagger_output = '0.RESULTS/emb_tagger'
    CRAFT_ONTOLOGIES = ['CHEBI', 'CL', 'GO', 'MONDO', 'MOP', 'NCBITAXON', 'PR', 'SO', 'UBERON']
    annotations_paths = glob.glob('0.RESULTS/preprocessing/*/annotations.csv')
    args = parse_args()
    print(args)
    timestamp = args.get('timestamp', get_now_str())

    # extract keywords bertopic
    print("Extracting keywords using BERTopic...")
    extracted_keywords_path_bertopic = os.path.join(results_path, 'bertopic', f'bertopic_keywords_{timestamp}.csv')
    if not os.path.exists(extracted_keywords_path_bertopic):
        extracted_keywords_path_bertopic, model_path_bertopic = get_keywords_bertopic(prepared_data_path, models_path,
                                                                                      results_path, timestamp)
        print(f"BERTopic model saved to {model_path_bertopic}")
        print(f"BERTopic extracted keywords saved to {extracted_keywords_path_bertopic}")
    else:
        print(f"Loaded extracted keywords from {extracted_keywords_path_bertopic}")

    # extract keywords lda
    print("Extracting keywords using LDA...")
    extracted_keywords_path_lda = glob.glob(os.path.join(results_path, 'lda', f'*{timestamp}*'))
    if not extracted_keywords_path_lda:
        num_topics = 7
        num_keywords = 22
        extracted_keywords_path_lda, model_path_lda = get_keywords_lda(prepared_data_path, models_path,
                                                                       os.path.join(results_path, 'lda'),
                                                                       timestamp, num_topics, num_keywords)
        print(f"LDA model saved to {model_path_lda}")
        print(f"LDA extracted keywords saved to {extracted_keywords_path_lda}")
    else:
        extracted_keywords_path_lda = extracted_keywords_path_lda[0]
        print(f"Loaded extracted keywords from {extracted_keywords_path_lda}")

    print('\nExtracting keywords done. Starting tagging.')

    print("Embedding tagger")
    print("Calculating embeddings of keywords...")
    keyword_files_input = {
        'bertopic': extracted_keywords_path_bertopic,
        'LDA': extracted_keywords_path_lda
    }

    keyword_emb_files_output = os.path.join(embeddings_path, f'{timestamp}_keywords_embeddings.csv')

    if not os.path.exists(keyword_emb_files_output):
        print("Calculating embeddings for keywords extracted")
        keyword_files_output = calculate_embeddings_for_keywords(
            [extracted_keywords_path_bertopic, extracted_keywords_path_lda],
            embeddings_path, timestamp)

        print(f"Embeddings are saved in {keyword_files_output}")
    else:

        print(f"Embeddings are loaded from {keyword_emb_files_output}")

    emb_files_output = glob.glob(os.path.join(emb_tagger_output, f'*{timestamp}*'))
    emb_files = glob.glob('0.RESULTS/embeddings/*labels.csv')
    if not emb_files_output:
        print("Embedding tagging")
        emb_files_output = emb_tagger(keyword_files_input,
                                      emb_files, keyword_emb_files_output, emb_tagger_output)
        print("Embedding tagging results saved in: ")
    else:
        print("Embedding tagging results leaded from: ")

    for file in emb_files_output:
        print(f"\t{file}")

    print("NCBO tagger for BERTopic")
    ncbo_bertopic_tagged_keywords_path = f'{results_path}/bertopic_ncbo/bertopic_ncbo_{timestamp}.csv'
    if not os.path.exists(ncbo_bertopic_tagged_keywords_path):
        ncbo_bertopic_tagged_keywords_path = tag_ncbo(CRAFT_ONTOLOGIES,
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
        ncbo_lda_tagged_keywords_path = tag_ncbo(CRAFT_ONTOLOGIES,
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

    bertopic_disambiguation_results_path = os.path.join(results_path, 'disambiguation',
                                                        f'bertopic_ncbo_{timestamp}_no_sorting_no_weighting_forcing.csv')
    if not os.path.exists(bertopic_disambiguation_results_path):

        bertopic_disambiguation_results_path = prepare_disambiguation(bertopic_disambiguation_results_path,
                                                                      ncbo_bertopic_tagged_keywords_path,
                                                                      extracted_keywords_path_bertopic,
                                                                      ncbo_emb_path, timestamp, forced=True)
        print(f"BERTopic disambiguation saved in {bertopic_disambiguation_results_path}")
    else:
        print(f"BERTopic disambiguation loaded from {bertopic_disambiguation_results_path}")

    lda_disambiguation_results_path = os.path.join(results_path,'disambiguation',
                                                   f'lda_ncbo_{timestamp}_no_sorting_no_weighting_forcing.csv')
    if not os.path.exists(lda_disambiguation_results_path):
        lda_disambiguation_results_path = prepare_disambiguation(lda_disambiguation_results_path,
                                                                 ncbo_lda_tagged_keywords_path,
                                                                 extracted_keywords_path_lda,
                                                                 ncbo_emb_path, timestamp, forced=True)
        print(f"LDA disambiguation saved in {lda_disambiguation_results_path}")
    else:
        print(f"LDA disambiguation loaded from {lda_disambiguation_results_path}")

    print("Saving summary")

    ontologies_mappings_paths = glob.glob('../0.RESULTS/preprocessing/_ontologies_mappings/*')

    sum_path = summary([bertopic_disambiguation_results_path, lda_disambiguation_results_path],
                       emb_files_output, annotations_paths, ontologies_mappings_paths, results_path, timestamp)
    print(f"Saved summary to {sum_path}")


def summary(disambiguation_results, emb_results, annotations_paths, ontologies_mappings_paths, results_path, timestamp):
    import pandas as pd
    import numpy as np
    import glob
    import os
    import copy
    def precision(TP, FP):
        return TP / (TP + FP)

    def recall(TP, FN):
        return TP / (TP + FN)

    def F1(precision, recall):
        return 2 / (1 / precision + 1 / recall)

    def return_TP_FN_FP(keywords, ground_truth):
        keywords = set(keywords)
        ground_truth = set(ground_truth)
        TP = len(keywords.intersection(ground_truth))
        FN = len(keywords.difference(ground_truth))
        FP = len(ground_truth.difference(keywords))
        return np.array([TP, FN, FP])

    def process_id(id_):
        if ':' in id_:
            id_ = id_.split(':')
        else:
            id_ = id_.split('_')
        try:
            id_[1] = int(id_[1])
        except (ValueError, IndexError) as e:
            pass
        return tuple(id_)

    annotations = []
    for file in annotations_paths:
        onto = os.path.basename(os.path.dirname(file))
        if 'GO_' in onto:
            onto = 'GO'
        tmp = pd.read_csv(file)
        tmp['ontology'] = onto.lower()
        annotations.append(tmp)
    annotations = pd.concat(annotations).drop(['Unnamed: 0', 'StartIndex', 'EndIndex'], axis=1) \
        .rename({'EntityID': 'id'}, axis=1)
    annotations['id'] = annotations['id'].apply(lambda x: x.lower()).apply(process_id)
    annotations_groupped = annotations.groupby('PMID').apply(lambda x:
                                                             pd.Series({'true_ids': set(x['id'])},
                                                                       index=['true_ids'])).reset_index()

    stats = []
    for file in emb_results:
        tagged = pd.read_csv(file)
        tagged['id'] = tagged['id'].apply(lambda x: x.lower()).apply(process_id)
        tagged_groupped = tagged.groupby('PMID').apply(lambda x:
                                                       pd.Series({'pred_ids': set(x['id'])},
                                                                 index=['pred_ids'])).reset_index()
        merged = tagged_groupped.merge(annotations_groupped, how='outer')

        TP_FN_FP_ids = np.zeros((3,))
        for i, row in merged.iterrows():
            TP_FN_FP_ids += return_TP_FN_FP(row['pred_ids'], row['true_ids'])

        precision_ = precision(TP_FN_FP_ids[0], TP_FN_FP_ids[2])
        recall_ = recall(TP_FN_FP_ids[0], TP_FN_FP_ids[1])
        stats.append((os.path.basename(file), precision_ * 100, recall_ * 100, 100 * F1(precision_, recall_)))
    pd.DataFrame(stats, columns=['file', 'precision', 'recall', 'F1'])

    ontologies_mappings = pd.concat(
        [pd.read_csv(file) for file in ontologies_mappings_paths])

    all_ncbo_concepts_used = pd.concat([pd.read_csv(file) for file in disambiguation_results])
    text_to_id = dict()
    for elem in sum(all_ncbo_concepts_used['ncbo_annotations'].apply(eval), []):
        id_ = elem['annotatedClass']['@id'].split('/')[-1]
        for annotation in elem['annotations']:
            text_to_id[annotation['text'].lower()] = id_

    ontologies_mappings['label'] = ontologies_mappings['label'].apply(lambda x: x.lower())
    label_to_ids = ontologies_mappings.groupby('label').apply(lambda x: list(set(x['id']))).reset_index().rename(
        {0: 'ids'}, axis=1)
    label_to_ids = dict(zip(label_to_ids['label'], label_to_ids['ids']))

    for file in disambiguation_results:
        dis = pd.read_csv(file)
        dis['disambiguation_best_concept'] = dis['disambiguation_best_concept'].apply(eval).apply(
            lambda x: [v[0].lower() for v in x.values()])
        dis = dis[['PMID', 'disambiguation_best_concept']].explode('disambiguation_best_concept') \
            .rename({'disambiguation_best_concept': 'label'}, axis=1)

        dis['id'] = dis['label'].map(label_to_ids)
        dis['id'] = dis['id'].apply(lambda x: x if x == x else [])

        dis['id'] = dis['id'].apply(lambda x: [process_id(elem.lower()) for elem in x])
        dis = dis.groupby('PMID').apply(lambda x:
                                        pd.Series({'pred_ids': set(sum(x['id'], []))},
                                                  index=['pred_ids'])).reset_index()
        merged = dis.merge(annotations_groupped, on=['PMID'], how='outer')

        TP_FN_FP_ids = np.zeros((3,))
        for i, row in merged.iterrows():
            TP_FN_FP_ids += return_TP_FN_FP(row['pred_ids'], row['true_ids'])

        precision_ = precision(TP_FN_FP_ids[0], TP_FN_FP_ids[2])
        recall_ = recall(TP_FN_FP_ids[0], TP_FN_FP_ids[1])
        stats.append((os.path.basename(file), precision_, recall_, F1(precision_, recall_)))
    results = pd.DataFrame(stats, columns=['file', 'precision', 'recall', 'F1'])

    output = os.path.join(results_path, f'results_{timestamp}.csv')
    results.sort_values('file').to_csv(output, index=False)
    return output


def calculate_embeddings_for_keywords(files, embeddings_output_filepath, timestamp):
    from emb_helpers import return_embeddings_for_concepts
    from transformers import AutoTokenizer, AutoModel
    import torch
    import pandas as pd
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    biobert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device_name)

    df = pd.concat([pd.read_csv(file) for file in files])
    all_keywords = sum(df['topic_keywords'].apply(eval), [])
    all_keywords = list(set([keyword for keyword, prob in all_keywords]))
    all_keywords = pd.DataFrame(all_keywords, columns=['keyword'])
    embeddings = return_embeddings_for_concepts(all_keywords['keyword'], biobert, 'cpu')
    embeddings['keyword'] = all_keywords['keyword']
    embeddings = embeddings[['keyword'] + list(range(len(embeddings.columns) - 1))]
    embeddings.to_csv(os.path.join(embeddings_output_filepath, f'{timestamp}_keywords_embeddings.csv'), index=False)
    return os.path.join(embeddings_output_filepath, f'{timestamp}_keywords_embeddings.csv')


def emb_tagger(emb_files_input, emb_files, embeddings_path, emb_tagger_output):
    import pandas as pd
    import numpy as np
    from numba import jit, njit, prange
    import glob
    import os
    import tqdm
    emb_cols = [str(col) for col in range(767 + 1)]

    @njit
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @njit(parallel=True)
    def return_cos_similarities(embedding, concept_embeddings):
        result = np.zeros((concept_embeddings.shape[0],))
        for i in prange(concept_embeddings.shape[0]):
            result[i] = cosine_sim(embedding, concept_embeddings[i])
        return result

    @njit
    def return_negative_distance(x, y):
        return -np.linalg.norm(x - y)

    @njit(parallel=True)
    def return_negative_distances(embedding, concept_embeddings):
        result = np.zeros((concept_embeddings.shape[0],))
        for i in prange(concept_embeddings.shape[0]):
            result[i] = return_negative_distance(embedding, concept_embeddings[i])
        return result

    proper_onto_names = {'chebi', 'cl', 'go', 'mop', 'ncbitaxon', 'pr', 'so', 'uberon'}

    def process_id(id_):
        if ':' in id_:
            id_ = id_.split(':')
        else:
            id_ = id_.split('_')
        try:
            id_[1] = int(id_[1])
        except (ValueError, IndexError) as e:
            pass
        return tuple(id_)

    # emb_files = [file for file in glob.glob(embeddings_output_filepath + '*') if
    #             'embeddings' not in os.path.basename(file)]

    keywords_emb = pd.read_csv(embeddings_path)

    def return_embedded_words(embedding, onto_emb, onto_emb_raw, similarity_func, n_words=1):
        similarities = similarity_func(embedding, onto_emb_raw)
        argsort_indices = np.argsort(similarities)
        chosen_concepts_names = set()
        i = 0
        while len(chosen_concepts_names) != n_words:
            max_indices = argsort_indices[-(n_words + i):]
            chosen_concepts_names = set(onto_emb.iloc[max_indices]['concept_name'])
            i += 1
        return max_indices, similarities[max_indices]

    similarity_funcs = {
        'distance': return_negative_distances,
        'cosine': return_cos_similarities
    }

    def tag_keywords_with_onto(onto_emb_file):
        onto_emb = pd.read_csv(onto_emb_file)
        onto_emb['id'] = onto_emb['id'].apply(lambda x: process_id(str(x).lower()))
        onto_emb = onto_emb.loc[onto_emb['id'].apply(lambda x: x[0] in proper_onto_names)]
        onto_emb['id'] = onto_emb['id'].apply(lambda x: '_'.join([str(elem) for elem in x]))

        onto_emb_raw = np.ascontiguousarray(onto_emb[emb_cols].values.astype(np.float32))
        onto_emb = onto_emb.drop(emb_cols, axis=1)
        tagged_keywords = []
        for i, row in keywords_emb.iterrows():
            keyword_emb = row[emb_cols].values.astype(np.float32)
            for similarity_func_name, similarity_func in similarity_funcs.items():
                indices, sims = return_embedded_words(keyword_emb, onto_emb, onto_emb_raw, similarity_func, n_words=5)
                concepts_names = onto_emb.iloc[indices]['concept_name'].to_list()
                concepts_ids = onto_emb.iloc[indices]['id'].to_list()
                tagged_keywords.append((
                    row['keyword'],
                    similarity_func_name,
                    concepts_names,
                    concepts_ids,
                    sims
                ))
        return pd.DataFrame(tagged_keywords,
                            columns=['keyword', 'similarity_type', 'concept_names', 'ids', 'similarities'])

    all_tagged_keywords = []
    for onto_file in emb_files:
        tagged_keywords = tag_keywords_with_onto(onto_file)
        tagged_keywords['ontology'] = os.path.basename(onto_file).replace('_labels.csv', '')
        all_tagged_keywords.append(tagged_keywords)
    all_tagged_keywords = pd.concat(all_tagged_keywords)

    def return_best_concept(x):
        result = {}
        max_mask = x['similarities'].values == x['similarities'].max()

        result['concept'] = x['concept_names'].values[max_mask]
        result['id'] = x['ids'].values[max_mask]
        result['ontology'] = tuple(x['ontology'].values[max_mask])

        return pd.Series(result, index=['concept', 'id', 'ontology'])

    best_concepts = all_tagged_keywords.explode(column=list(all_tagged_keywords.columns[2:-1])) \
        .groupby(['keyword', 'similarity_type']).apply(return_best_concept).reset_index() \
        .explode(['concept', 'id']).drop_duplicates()

    output_filenames = []
    for name, filename in emb_files_input.items():
        for sim_type in ['cosine', 'distance']:
            keywords_df = pd.read_csv(filename)
            basename = os.path.basename(filename).replace('.csv', '')

            keywords_df['topic_keywords'] = keywords_df['topic_keywords'].apply(eval)
            keywords_df['keyword'] = keywords_df['topic_keywords'].apply(lambda x: [keyword for keyword, prob in x])
            keywords_df = keywords_df.drop(['Unnamed: 0', 'topic_number', 'topic_probs', 'topic_keywords'], axis=1,
                                           errors='ignore')
            keywords_df = keywords_df.explode('keyword')
            keywords_df = keywords_df.merge(best_concepts[best_concepts['similarity_type'] == sim_type], how='left')
            # print(keywords_df.count())
            method = 'bertopic' if 'bertopic' in filename else 'lda'
            output_filename = os.path.join(emb_tagger_output, f'{basename}_{method}_{sim_type}_tagged.csv')
            output_filenames.append(output_filename)
            keywords_df.to_csv(output_filename, index=False)

    return output_filenames


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
    embeddings = return_embeddings_for_concepts(all_words_to_embed['words'], biobert, device_name)
    embeddings['words'] = all_words_to_embed['words']

    basename = f'{timestamp}_ncbo_embeddings.csv'
    output_path = os.path.join(embeddings_output_filepath, basename)
    embeddings.to_csv(output_path, index=False)
    return output_path


if __name__ == '__main__':
    main()
