import pandas as pd
import copy
import math
import numpy as np
import tqdm
import datetime
import os


def get_embeding(word, emb):
    """Get word embedding """
    return emb.loc[word]


def create_tags_list_dict(row, with_sorting=False):
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
            if value not in result[key]:
                result[key][value] = 1
            else:
                result[key][value] += 1
        else:
            result[key] = {value: 1}
    if with_sorting:
        for key in result.keys():
            d = result[key]
            d = {k: 0 for k, v in dict(sorted(d.items(), key=lambda item: item[1], reverse=True)).items()}
            result[key] = d
    else:
        for key in result.keys():
            d = result[key]
            d = {k: 0 for k, v in d.items()}
            result[key] = d

    return result


def disambiguation(current_selection, embedings, weigths, forced):
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
    new_current_selction = copy.deepcopy(current_selection)
    should_stop = False
    for i in range(7):

        for keyword, concepts_list in new_current_selction.items():
            distances = {}  # for each possible concept calaculate the mean distance from other kewords (concepts of them)
            if forced and any([c == keyword for c in concepts_list.keys()]):
                _ = [c != keyword for c in concepts_list.keys()]
                distances = dict(zip(_, [1000] * len(_)))
                distances[keyword] = 0
            else:
                for concept in concepts_list.keys():
                    distances[concept] = []
                    for k, current_best_tags in new_current_selction.items():
                        # foreach keyword that is not a current one 
                        if k != keyword:
                            current_best_tag = list(current_best_tags.keys())[0]  # the first out of list of concepts
                            try:
                                distances[concept].append(weigths[k] * math.dist(get_embeding(concept, embedings),
                                                                                 get_embeding(current_best_tag,
                                                                                              embedings)))  # append distance from this concept
                            except Exception as e:
                                print(e)
                    distances[concept] = np.mean(distances[concept])  # mean distance
            new_current_selction[keyword] = dict(
                sorted(distances.items(), key=lambda item: item[1]))  # upadate the current selection of this keyword
    return new_current_selction


def keywords_importance(grouped_data, tagger_data):
    """get keywords importance"""
    return grouped_data.reset_index().merge(tagger_data[['PMID', 'topic_keywords']], on='PMID').set_index(
        'text_to_annotate')


def get_n_best_tags(data, n=1):
    """how many best concepts to take"""
    return [{k: sorted(v, key=v.get)[:n] for k, v in dd.items()} for dd in data['after_disambiguation']]


def prepare_for_disambiguation(data, tagger, embedings, column_name='ncbo_annotations_pairs', weighting=False,
                               sorting=False, forced=False, take_best=1):
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
    grouped = keywords_importance(grouped, tagger)
    grouped['possible_tags'] = grouped[column_name].apply(lambda r: create_tags_list_dict(r, sorting))

    # disambiguation
    res = []
    for idx, row in tqdm.tqdm(grouped.iterrows(), total=len(grouped)):
        current_selection = row['possible_tags']
        if not weighting:
            weigths = dict(zip(list(row['topic_keywords'].keys()), [1] * len(row['topic_keywords'])))
        else:
            weigths = row['topic_keywords']
        r = disambiguation(current_selection, embedings, weigths, forced)
        res.append(r)
    grouped['after_disambiguation'] = res
    data = data.merge(grouped['after_disambiguation'].reset_index(), on='text_to_annotate')
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

    data = pd.read_csv(data_name)
    data['ncbo_annotations_pairs'] = data['ncbo_annotations_pairs'].apply(eval)
    data['ncbo_annotations_pairs'] = data['ncbo_annotations_pairs'].apply(lambda x: [[a[0].upper(), a[1]] for a in x])

    tagger = pd.read_csv(tagger_name)
    tagger['topic_keywords'] = tagger['topic_keywords'].apply(eval).apply(
        lambda x: {k.upper(): v for k, v in dict(x).items()})

    embedings = pd.read_csv(embedings_name)
    embedings = embedings.set_index('words')
    embedings.index = embedings.index.str.upper()
    embedings = embedings[~embedings.index.duplicated(keep='first')]

    return data, tagger, embedings


def prepare_disambiguation(results_folder, data_path, tagger_path, embeddings_path, timestamp, weigthing=False,
                           sorting=False, forced=False):
    """Performs disambiguation

    Parameters
    ----------
    results_folder : str
        Path to save results
    
    data_path : str
        Path to save the model to (folder must exist).

    tagger_path : str
        Path to save the results to (folder must exist).

    embeddings_path : str
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
    import copy
    import pandas as pd
    import os
    import math
    import numpy as np

    data, tagger, embeddings = prepare_data(data_path, tagger_path, embeddings_path)
    data = prepare_for_disambiguation(data, tagger, embeddings, 'ncbo_annotations_pairs', weigthing, sorting, forced)
    data.to_csv(results_folder)

    return results_folder


def get_lda_results(data_path, num_topics=10, num_keywords=10):
    from gensim import corpora, models
    import os
    import pandas as pd
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
        for t in lda_model.print_topics(number_of_topics, number_of_keywords):
            topics_distrib[t[0]] = [(a.split('*')[1][1:-1], float(a.split("*")[0])) for a in t[1].split(' + ')]
        return topics_distrib

    data = pd.read_csv(data_path)
    columns = ['tokenized_sentences', 'tokenized_words_lemmatize']
    for col in columns:
        data[col] = data[col].apply(eval)

    texts = data.groupby('PMID')['tokenized_words_lemmatize'].agg(lambda x: x.iloc[0] + x.iloc[1])
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = models.LdaMulticore(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=num_topics,
                                    passes=20)
    doc_lda = lda_model[corpus]

    topic_distribution = get_topic_distribution(lda_model, num_topics, num_keywords)
    topics_results = pd.DataFrame.from_records([topic_distribution]).T.reset_index().rename(
        columns={'index': 'topic_number', 0: 'topic_keywords'})
    topics_results['keywords'] = topics_results['topic_keywords'].apply(lambda x: [a[0] for a in x])

    docs = []
    for doc in doc_lda:
        docs.append({
            'topic_number': doc[0][0],
            'topic_probs': float(doc[0][1]),
            'topic_keywords': topics_results.iloc[doc[0][0]]['topic_keywords'],
            'keywords': topics_results.iloc[doc[0][0]]['keywords']

        })

    docs = pd.DataFrame.from_records(docs)

    results = data[['PMID']].drop_duplicates().reset_index(drop=True).join(docs)
    topics_results = pd.DataFrame.from_records([topic_distribution]).T.reset_index().rename(
        columns={'index': 'topic_number', 0: 'topic_keywords'})

    return results, topics_results, lda_model


def get_keywords_lda(data_path, models_path, results_path, timestamp, num_topics=10, num_keywords=10):
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

    results, topics_results, lda_model = get_lda_results(data_path, num_topics, num_keywords)

    results_path = os.path.join(os.path.join(results_path, f'lda_results_{timestamp}.csv'))
    results.to_csv(results_path)

    models_path = os.path.join(models_path, f"lda_model_{timestamp}")
    lda_model.save(models_path)

    return results_path, models_path

if __name__ == '__main__':
    # Data preperation
    MAX_ITER = 7
    data_name = '../0.RESULTS/lda_ncbo/lda_ncbo_2023-01-19_20-55-05_keywords_22_topics_6.csv'
    tagger_name = '../0.RESULTS/lda/lda_results_2023-01-19_20-01-44_keywords_22_topics_7.csv'
    embeding_name = '../0.RESULTS/embeddings/ncbo_embeddings_19_01_23_22_0_0.csv'
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    weigthing = True
    sorting = True
    forced = False

    keyword_extraction_method = 'bertopic'
    tagger = 'ncbo'
    keywords = 22
    topics = 7
    min_topic_size = 6
    args = f'_keywords_{keywords}_topics_{topics}'
    results_folder = f'../0.RESULTS/disambiguation/{keyword_extraction_method}{args}_{tagger}_{timestamp}_no_sorting_no_weighting_no_forcing.csv'

    prepare_disambiguation(results_folder, data_name, tagger_name, embeding_name, timestamp, weigthing=False, sorting=False,
                           forced=False)
