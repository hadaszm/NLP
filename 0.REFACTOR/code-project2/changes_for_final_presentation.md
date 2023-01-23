# Changes from POC2 to final presenatation (POC2 changes can be seen in file changes_for_poc2.md):

* 1.DATA_PREPARATION 

* 2.KEYWORDS_EXTRACTION
    * bert_topic - testing different values of hyperparametrs, refactored into one function.
    * LDA - adjusted number of topics based on hyperparametertunning results. Added analysis and refactored code
* 3.TAGGING
    * bert_embeddings - 
    * emb_tagger - 
    * ncbo tagger - used for CRAFT tagging with the best hyperparams values
* 4.DISAMBIGUATION - added forcing option. Added analysis
* 5.EVALUATION - 
    * keywords_extraction - file including all evaluation processes concerning BERTopic and LDA, namely: defining metrics for comparing with CRAFT ground-truth mentions, qualitative analysis of obtained results of BERTopic and LDA, hyperparams tests and compariosons


