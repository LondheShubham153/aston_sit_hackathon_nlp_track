from preprocess_tweets import PreProcessor
from vectorize_tweets import Vectorizer

class SentimentAnalyzer:

  def analyse(self,model,input_text):
    """
    Pre Processing
    BOW transformation
    TFIDF transformation

    """
    processor = PreProcessor()
    vectorizer = Vectorizer()
    final_word_vocab,bag_of_words = vectorizer.vectorize(processor)
    tfIdf_obj = vectorizer.create_tfidf(bag_of_words)

    pre_processed_features = processor.process(input_text)
    bow_feature = final_word_vocab.transform(pre_processed_features)
    tfIdf_feature = tfIdf_obj.transform(bow_feature)
    return max(set(model.predict(tfIdf_feature)), key = list(model.predict(tfIdf_feature)).count)

