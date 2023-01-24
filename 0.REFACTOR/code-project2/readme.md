* 0.RESULTS - folder with results from each step
    * bertopic
    * bertopic_ncbo
    * disambiguation - results of the disambiguation
    * embedings
    * lda - results of the lda keywords extraction
    * lda_ncbo
    * preprocessing - the data retrieved in CRAFT dataset and ontologies preperation step
        * _ontologies_mapping - folder with mappings concept_id:concept_name
            * ONTOLOGY_NAME_labels.csv - csv file containing mappings from a given ontology
        * ONTOLOGY_NAME - folder specific name for each ontology
            * annotations.csv - annoations present in CRAFT dataset from this ontology 
        * data_whole.csv - preprocessed CRAFT data (lemmatization, tokenization etc.)
    
    ATTENTION: this folder (0.RESULTS) was to big for github, hence it is on the drive - link https://drive.google.com/drive/folders/1eeKlWWfk3WOvIMi3mc2QTX3d9LUaCriY

* 1.DATA_PREPARATION 
    * CRAFT
        * articles - folder with CRAFT articles (txt without annotations) - part of CRAFT dataset
        * concept-annotation - folder contains folders with the ontologies names. In each folder there are annotations with given ontlogy - part of CRAFT dataset
        * data_analysis.ipynb - data preprocessing using CRAFT. On output data used in the next steps is saved 
            0.RESULTS/preprocessing/ONTOLOGY_NAME and 0.RESULTS/preprocessing/data_whole.csv
    * ontologies
        * ontologies_mapping_preparation.ipynb - file contaiing the processing of ontologies .owl files to the form of concept_id:concpet_name, results saved in 0.RESULTS/preprocessing/-ontologies_mapping

* 2.KEYWORDS_EXTRACTION
    * bert_topic
        * bert_topic.ipynb - file containing implementation for BERTopic keywords extraction
    * LDA
        * LDA_analysis.ipynb - keyword extraction using LDA
* 3.TAGGING
    * bert_embeddings
        * emb_helpers.py - file with functions that are used in notebooks in this folder, they are used to calculate embedding 
        * saving_embeddings_from_concepts_biobert_craft.ipynb - calculate embeddings from concepts' names from ontologies (very long)
        * saving_embeddings_from_keywords_biobert_craft.ipynb - calculate embeddings from keywords from LDA and BERTopic
        * saving_embeddings_from_ncbo_biobert_craft.ipynb - calculate embeddings from tags from NCBO
    * emb_tagger
        * tagging_craft.ipynb - taggs keywords from LDA and BERTopic with concepts from ontologies 
    * ncbo tagger
        * ncbo_tagger.ipynb - file containing NCBO tagging proccess (based either on bertopic keywords or NCBO keywords)
* 4.DISAMBIGUATION
    * disambiguation.ipynb - script for performing disambiguation
* 5.EVALUATION
    * stats_craft.ipynb - calculates F1 score, recall and precision from e2e solution

changes_for_poc2.md - what did we implement up to the poc2.
changes_for_final_presentation.md - what did we implement from poc2 until the final project version.
<hr style="border:2px solid gray">
Steps to reproduce the results:

1. Download `reproduce_script.rar` from https://drive.google.com/drive/folders/1-lAjO3VyEMt0KWbpxhLRIx_L7h_SIuaZ?usp=sharing
2. `cd reproduce_script`
3. Run script with: 
	`python script.py` <br />
	If you want to stop the program and run it later you can use timestamp from filenames of results for example:  <br />
	`python script.py timestamp=2023-01-24_16-21-53`
4. results will appear in 0.RESULTS folder 