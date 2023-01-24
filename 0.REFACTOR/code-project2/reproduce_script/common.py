import sys


def get_now_str():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def transform_strings_to_arrays(df, col_names=['tokenized_sentences', 'tokenized_words', 'tokenized_words_processed',
                                               'tokenized_words_no_stopwords', 'tokenized_words_lemmatize']):
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
