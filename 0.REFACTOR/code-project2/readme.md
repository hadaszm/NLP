0.RESULTS - folder with results from each step
    * bertopic
    * bertopic_ncbo
    * disambiguation
    * embedings
    * lda
    * lda_ncbo
    * preprocessing - the data retrieved in CRAFT dataset and ontologies preperation step
        * _ontologies_mapping - folder with mappings concept_id:concept_name
            * ONTOLOGY_NAME_labels.csv - csv file containing mappings from a given ontology
        * ONTOLOGY_NAME - folder specific name for each ontology
            * annotations.csv - annoations present in CRAFT dataset from this ontology 
        * data_whole.csv - preprocessed CRAFT data (lemmatization, tokenization etc.)

1.DATA_PREPARATION 
    * concepts_from_umls
        * save_concepts.ipynb
    * CRAFT
        * articles - folder with CRAFT articles (txt without annotations) - part of CRAFT dataset
        * concept-annotation - folder contains folders with the ontologies names. In each folder there are annotations with given ontlogy - part of CRAFT dataset
        * data_analysis.ipynb - data preprocessing using CRAFT. On output data used in the next steps is saved 
    * ontologies
        * ontologies_mapping_preparation.ipynb - file contaiing the processing of ontologies .owl files to the form of concept_id:concpet_name, results saved in 0.RESULTS/preprocessing/-ontologies_mapping

2.KEYWORDS_EXTRACTION
    * bert_topic
        * bert_topic.ipynb - file containing implementation for BERTopic keywords extraction
    * LDA
        * LDA_analysis.ipynb - keyword extraction using LDA
3.TAGGING
    * bert_embeddings
        * emb_helpers.py
        * saving_embeddings_from_concepts_biobert_craft.ipynb
        * saving_embeddings_from_keywords_biobert_craft.ipynb
        * saving_embeddings_from_ncbo_biobert_craft.ipynb
    * emb_tagger
        * tagging.ipynb
        * tagging_colab.ipynb
        * tagging_craft.ipynb
    * ncbo tagger
        * ncbo_tagger.ipynb - file containing NCBO tagging proccess (based either on bertopic keywords or NCBO keywords)
4.DISAMBIGUATION
    * disambiguation.ipynb - di
5.EVALUATION
    * stats.ipynb
    * stats_craft.ipynb