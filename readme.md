Steps to reproduce the results:

1. Data download and preprocessing (TODO GOSIA)
    1.1 UMLS 
    Firstly ask for access https://www.nlm.nih.gov/research/umls/index.html, when granted download 2017AA and newest release. From 2017AA release copy MRSTY.RRF and MRCONSO_HISTORY.txt from newest release copy MRCONSO.RRF file. Run file concepts_from_umls/save_concepts.ipynb.
    1.2 Embeddings
    Run bert_embeddings/saving_embeddings_from_concepts_biobert.ipynb

2. LDA (TODO: GOSIA)

3. BERTopic 
Go to bert_topic folder and then to bert_topic.ipynb file. Choose the way you want to preprocess the data by runing cells in appropriate sections:
* Processed data (with steming and stop words removal)
* Raw data (without stemming and stopwords removal)
* Data with stopwords deleted (but without stemming performed)
* Data after lemmatization and with stopwords deleted
We recommend using 'Data after lemmatization and with stopwords deleted'.
Transformed data (kyewords found for each paper) will be saved in paths specified.  

4. NCBO Tagger
Go to ncbo_tagger folder and then to ncbo_tagger.ipynb file. In the first cell in Definitions section, specify the train_path, test_path and results_folder for the files used in this step (test and train data should be generated by LDA or BERTTopic in previous steps).
Run next cells. To connect to NCBO Api you need a valid api key. You can crete your own or (for the purpose of this project) use ours, specified in the 'NCBO tagger API conection' section.
Choose the way you want to annotate the papers with NCBO. Sections names decribe the way the annotation is made.
Run 'Save results' section to save the annotation results to files.
Section 'Results analysis' is dedicated for some basic results investigation.
Then, extract keywords-annotation matching with the use of previously generated files (section 'Extracting first order keywords-anotations matching') 
To save final annotation result run cells in section 'Save annotated pairs'

5. Embedding tagger
Run emb_tagger/tagging_colab.ipynb file with proper filenames as input  variable (emb_files)
6. Disambiguation (TODO GOSIA)

7. Metrics
Run summary/stats.ipynb file

