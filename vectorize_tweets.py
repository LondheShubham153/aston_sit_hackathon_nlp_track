from select import select
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from dataset_loader import DataCleaner

class Vectorizer:

    def vectorize(self,processor):
        """
        Build Vocabulary
        """
        word_vector = CountVectorizer(analyzer=processor.process)
        dataset = DataCleaner().load_data('dataset/final_cleaned_data.csv')
        features = dataset.iloc[:,1].values
        final_word_vocab = word_vector.fit(features)
        return (final_word_vocab,final_word_vocab.transform(features))
    
    def create_tfidf(self,bag_of_words):
        """
        Calculating TF and IDF
        and transform data by calculating Weights
        """
        return TfidfTransformer().fit(bag_of_words)


    