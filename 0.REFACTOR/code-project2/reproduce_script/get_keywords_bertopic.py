from common import transform_strings_to_arrays


def get_keywords_bertopic(data_path, models_path, results_path, timestamp, min_topic_size=6, top_n_words=22):
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

    min_topic_size: int
        minimal number of datapoints in topic

    top_n_words: int
        number of extracted keywords

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
        topic_model = BERTopic(min_topic_size=min_topic_size, top_n_words=top_n_words)
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

    ################################################################################################################

    full_data = transform_strings_to_arrays(pd.read_csv(data_path), col_names=['tokenized_words_lemmatize'])

    data = full_data.groupby(by=['PMID'])['tokenized_words_lemmatize'].agg(
        lambda x: ' '.join(x.values[0] + x.values[1]))

    model_name = f'bertopic_keywords_{timestamp}'
    model_save_name = os.path.join(models_path, model_name)
    result_path = os.path.join(results_path, 'bertopic', f'{model_name}.csv')

    topic_model, topics, probs = train_transform_save(data, model_save_name, min_topic_size=3)
    _, res_df = load_transform_save(data, model_save_name, result_path)

    return result_path, model_save_name
