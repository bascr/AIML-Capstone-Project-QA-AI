from yake import KeywordExtractor
from nltk.tokenize import sent_tokenize


class KeyConceptExtractor:

    def __init__(self, config):
        self.config = config
        self.kwe = KeywordExtractor(n=self.config["key_extractor"]["max_ngram_size"],
                                    dedupLim=self.config["key_extractor"]["deduplication_threshold"],
                                    dedupFunc=self.config["key_extractor"]["deduplication_algo"],
                                    top=self.config["key_extractor"]["numOfKeywords"])

    def get_key_words(self, doc):
        tokenized_sentences_lst = sent_tokenize(doc)
        sentences_lst = []
        for sentence in tokenized_sentences_lst:
            sentence_keywords = []
            keywords_lst = self.kwe.extract_keywords(sentence)
            for keyword in keywords_lst:
                sentence_keywords.append(keyword)
            sentences_lst.append((sentence, sentence_keywords))
        return sentences_lst
