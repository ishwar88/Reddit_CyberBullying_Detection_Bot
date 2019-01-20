# Data Processing Tools
import praw
import pickle
import pandas
import numpy

# Machine Learning Tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Natural Language Processing Tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

class CyberbullyingDetectionEngine:
    """ Class that deals with training and deploying cyberbullying detection models
    """

    def __init__(self):
        self.corpus = None
        self.tags = None
        self.lexicon = None
        self.vectorizer = None
        self.model = None
        self.metrics = None

    class CustomVectorizer:
        """ Extracts features from text and creates word vectors
        """
        def __init__(self, lexicons):
            self.lexicons = lexicons

        def transform(self, corpus):
            word_vectors = []
            for text in corpus:
                features = []
                for k, v in self.lexicons.items():
                    features.append(len([w for w in word_tokenize(text) if w in v]))

                word_vectors.append(features)

            return numpy.array(word_vectors)

    def _simplify(self, corpus):
        """ Takes in a list of strings (corpus and remove stopwords
            (I, a, an, the, of, etc.) and convert to lowercase,
            remove non-alphanumeric characters, stem (removing tense from words)
        """

        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')

        def clean(text):
            text = re.sub('[^a-zA-Z0-9]', ' ', text)
            words = [stemmer.stem(w) for w in word_tokenize(text.lower()) if w not in stop_words]
            return " ".join(words)
            

        return [clean(text) for text in corpus]

    def _model_metrics(self, features, tags):
        """ Takes in testing data and returns a dictionary of metrics
        """
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        predictions = self.model.predict(features)
        
        for r in zip(predictions, tags):
            if r[0] == 1 and r[1] == 1:
                tp += 1
            elif r[0] == 1 and r[1] == 0:
                fp += 1
            elif r[0] == 0 and r[1] == 1:
                fn += 1
            elif r[0] == 0 and r[1] == 0:
                tn += 1
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return {
            'precision': precision,
            'recall': recall,
            'f1' : (2*precision*recall)/ (precision + recall)
        }
    def _get_lexicon(self,path):
        """ Takes in a path to a text file and returns a set
            set containing every word in the file
        """
        words = set()
        with open(path) as file:
            for line in file:
                words.update(line.strip().split(' '))

        return words

    
    def load_corpus(self, path, corpus_col, tag_col):
        """ Takes in a path to pickled pandas dataframe, the name of the corpus column,
            and the name of the tag column, and extracts a tagged corpus
        """

        data = pandas.read_pickle(path)[[corpus_col, tag_col]].values
        self.corpus = [row[0] for row in data]
        self.tags = [row[1] for row in data]

    def load_lexicon(self, fname):
        """ Loads a set of words from a text file
        """
        if self.lexicon is None:
            self.lexicon = {}

        self.lexicon[fname] = self._get_lexicon('./data/' + fname + '.txt')
        
        

    def load_model(self, model_name):
        """ Loads a ML model, its corresponding vectorizer, and its performance metrics
        """
        self.model = pickle.load(open('./models/' + model_name + '_ml_model.pkl', 'rb'))
        self.vectorizer = pickle.load(open('./models/' + model_name + '_vectorizer.pkl', 'rb'))
        self.metrics = pickle.load(open ('./models/' + model_name + '_metrics.pkl', 'rb'))

    def train_using_bow(self):
        """Trains a model using a Bag of Words (word counts) on the corpus and tags them
        """

        corpus = self._simplify(self.corpus)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

        bag_of_words = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(bag_of_words, self.tags, test_size = 0.2, stratify = self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    def train_using_custom(self):
        """ Trains model using a custom feature extraction approach
        """

        corpus = self._simplify(self.corpus)
        self.vectorizer = self.CustomVectorizer(self.lexicon)

        word_vectors = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(word_vectors, self.tags, test_size = 0.2, stratify = self.tags)
    
        self.model = SVC()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)
            
    def train_using_tfidf(self):
        """ Trains model using tf-idf weighted word counts as features
        """

        corpus = self._simplify(self.corpus)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)

        word_vectors = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(word_vectors, self.tags, test_size = 0.2, stratify = self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)
                                                            
        

    def predict(self, corpus):
        """ Takes in a text corpus and returns predictions
        """

        x = self.vectorizer.transform(self._simplify(corpus))
        return self.model.predict(x)

    def evaluate(self):
        """ Return model metrics
        """

        return self.metrics

    def save_model(self, model_name):
        """ Saves the model for future use
        """
        pickle.dump(self.model, open('./models/' + model_name + '_ml_model.pkl', 'wb'))
        pickle.dump(self.vectorizer, open('./models/' + model_name + '_vectorizer.pkl', 'wb'))
        pickle.dump(self.metrics, open ('./models/' + model_name + '_metrics.pkl', 'wb'))
    

if __name__ == '__main__':
    reddit = praw.Reddit(
        client_id = 'SNwAoH0So3RACg',
        client_secret = 'cb9sUubeppqw24Hb0Nc5B9D47QM',
        user_agent = 'Cyberbullying Detection Engine'
    )

    new_comments = reddit.subreddit('TwoXChromosomes').comments(limit=1000)
    queries = [comment.body for comment in new_comments]
    print(len(queries))
    engine = CyberbullyingDetectionEngine()
    engine.load_corpus('./data/final_labelled_data.pkl', 'tweet', 'class')
    
    """engine.train_using_bow()
    engine.save_model('bow')
    print(engine.evaluate())

    print(engine.predict(queries))

    engine.train_using_tfidf()
    engine.save_model('tfidf')
    print(engine.evaluate())

    print(engine.predict(queries))"""

    engine.load_lexicon("hate-words")
    engine.load_lexicon("neg-words")
    engine.load_lexicon("pos-words")

    engine.load_model('custom')
    """engine.train_using_custom()"""
    engine.save_model('custom')
    print(engine.evaluate())
    print(engine.predict(queries))
    
        

