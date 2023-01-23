* 1.DATA_PREPARATION 
    * CRAFT - new dataset - whole new analysis, tokenization, lemmatization and preprocessing was done. The data are formatted to the format used in MedMentions.
            Due to that reusing of already-prepered functions is possible.
    * ontologies - methods for anaylizing and tranforming ontologies used in CRAFT dataset

* 2.KEYWORDS_EXTRACTION
    * bert_topic - adjusted training process for CRAFT dataset 
    * LDA - adjusted number of topics based on bert_topic results
* 3.TAGGING
    * bert_embeddings - calculated embeddings for concept from ontologies used in CRAFT
    * emb_tagger - tagged keywords from LDA and BERTopic
    * ncbo tagger - adjusted ontologies queired from API based on the ones that are in CRAFT dataset 
* 4.DISAMBIGUATION - new options for disambiguation enabled. Initial sorting based on the concepts frequency and weighted voting based on the keywords importance.
* 5.EVALUATION - calculated metrics using annotations from CRAFT

What is more, we introduced during the call the following ideas for the final project:
* another disambiguation technique - based on forcing certain annotations, where annotation matches exacltly the keywords
* new evaluation techniques, aiming to validate only single parts of the solution (mainly keywords extraction), instead of the entire pipeline at once
* checking the influence of keywords extraction hyperparameters tuning

