0.RESULTS - folder with results from each step
    preprocessing - the data retrieved in data preperation step
            ONTOLOGY_NAME - specific name for each ontology
                annotations.csv - annoations from this ontlogy 
            data_whole.csv - preprocessed data (lemmatization, tokenization etc.)
    bertopic
    bertopic_ncbo
    disambiguation
    embedings
    lda
    lda_ncbo

1.DATA_PREPARATION 
    concepts_from_umls
        save_concepts.ipynb
    CRAFT
        articles - folder with articles (txt without annotations)
        concept-annotation - folder contains folders with the ontologies names. In each folder there are annotations with given ontlogy  
        data_analysis.ipynb - data preprocessing using CRAFT. On output data used in the next steps is saved 
    ontologies_analysis
        analysis.ipynb

2.KEYWORDS_EXTRACTION
    bert_topic
        bert_topic.ipynb
    LDA
        LDA_analysis.ipynb - keyword extraction using LDA
3.TAGGING
    bert_embeddings
        emb_helpers.py
    emb_tagger
        saving_embeddings_from_concepts_biobert_craft.ipynb
        saving_embeddings_from_keywords_biobert_craft.ipynb
        saving_embeddings_from_ncbo_biobert_craft.ipynb
    nbco_tagger
        tagging.ipynb
        tagging_colab.ipynb
        tagging_craft.ipynb
4.DISAMBIGUATION
    disambiguation.ipynb - di
5.EVALUATION
    stats.ipynb
    stats_craft.ipynb