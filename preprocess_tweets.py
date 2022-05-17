from nltk.corpus import stopwords
import string

class PreProcessor:
  
  def remove_punctuation(self, record):
    cleaned_str = [char for char in record if char not in string.punctuation]
    return ''.join(cleaned_str)
  
  def normalize_sentences(self, sentences):
    words = sentences.split(" ")
    return [word.lower() for word in words]

  def remove_stopwords(self,words):
    return [word for word in words if word not in stopwords.words("english")]

  def process(self, record):
      # Remove Punctuation
      sentences = self.remove_punctuation(record)
      
      # Normalize
      norm_words = self.normalize_sentences(sentences)
      
      # Remove Stopwords
      final_words = self.remove_stopwords(norm_words)
      
      return final_words