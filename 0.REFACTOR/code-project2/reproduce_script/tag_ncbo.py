from common import transform_strings_to_arrays


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
        data = transform_strings_to_arrays(data, col_names=['topic_keywords'])
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
        res_ann_pairs = []
        for _, concept in enumerate(found_concepts):
            max_trials = 5
            trials_no = 0
            while trials_no < max_trials:
                try:
                    concept_class = get_json(concept["annotatedClass"]["links"]["self"])
                    concept_class_ancestors = get_json(concept["annotatedClass"]['links']['ancestors'])
                    break
                except:
                    trials_no += 1
                    continue
            if trials_no == max_trials:
                raise Exception("number of unsuccessfull connection attempts is max_trials")
            annotations = concept['annotations']
            # annotations for this class
            for annot in annotations:
                res_ann_pairs.append(
                    [annot['text'], concept_class["prefLabel"], 'DIRECT', concept["annotatedClass"]["links"]["self"]])
            # annotations for ancestors
            for annot in annotations:
                for ancestor in concept_class_ancestors:
                    res_ann_pairs.append([annot['text'], ancestor["prefLabel"], 'ANCESTOR',
                                          concept["annotatedClass"]['links']['ancestors']])
        unique_ann_pairs = [list(x) for x in set(tuple(x) for x in res_ann_pairs)]
        return unique_ann_pairs

    ##########################################################################################################################

    # read data
    data = read_keywords_extraction_results(extracted_keywords_path)

    # annotate data
    data['ncbo_annotations'] \
        = data.text_to_annotate.apply(lambda text: \
                                          get_json(
                                              REST_URL + f"/annotator?ontologies={','.join(ontologies)}&text=" + urllib.parse.quote(
                                                  text)))

    data = data.reset_index()[['PMID', 'text_to_annotate', 'ncbo_annotations']]

    data_to_annotate = data[['text_to_annotate', 'ncbo_annotations']]
    data_to_annotate = data_to_annotate.loc[data_to_annotate.astype(str).drop_duplicates().index]
    data_to_annotate['ncbo_annotation_pairs'] = data_to_annotate.apply(create_annotation_pairs,
                                                                       column_name='ncbo_annotations', axis=1)

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
