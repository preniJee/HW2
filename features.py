"""
    Basic feature extractor
"""
from operator import methodcaller
import string
# from tf_idf import Tf_idf
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()


def preprocess(text, stopwords_file):
    # make lower case
    lower_cased_text = []
    for line in text:
        # print(line)
        # lower_cased_text.append(np.char.lower(line))
        lower_cased_text.append([str.lower(word) for word in line])
    #     print("after lower case")
    #     print(lower_cased_text[0])

    # remove stopwords
    without_stopwords_text = remove_stopwords(lower_cased_text, stopwords_file)

    #     print("after removing stopwprds")
    #     print(without_stopwords_text[0])

    # remove punctuation and single char words
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    without_symbols_text = []

    for sample in without_stopwords_text:
        without_symbol = [word for word in sample if word not in symbols and len(word) > 1]
        without_symbols_text.append(without_symbol)
    #     print(len(sample))
    #     print(len(without_symbol))
    preprocessed_text = without_symbols_text
    return preprocessed_text


def remove_stopwords(text, stopwords_file):
    with open(stopwords_file, "r") as f:
        stopwords = f.readlines()
    stripped_stopwords = []
    for word in stopwords:
        stripped_stopwords.append(word.strip())

    stopwords_removed = []
    for sample in text:
        without_stopwords = [word for word in sample if not word in stripped_stopwords]
        stopwords_removed.append(without_stopwords)
    return stopwords_removed


class Features:

    def __init__(self, data_file, stopwords_file):
        with open(data_file, "r", encoding='utf8') as file:
            data = file.read().splitlines()

        data_split = map(methodcaller("rsplit", "\t", 1), data)
        self.dataset_file = data_file
        texts, self.labels = map(list, zip(*data_split))

        self.numeric_classes = []
        unique_labels = list(np.unique(self.labels))

        for i, y in enumerate(self.labels):
            self.numeric_classes.append(unique_labels.index(y))

        print("Start preprocessing...")
        self.tokenized_text = [tokenize(text) for text in texts]

        self.labelset = list(set(self.labels))
        if not "odia" in data_file:
            self.preprocessed_text = preprocess(self.tokenized_text, stopwords_file)
        else:
            ## if the language is not english skip other preprocesing step and only remove punctuation and single char words
            symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
            without_symbols_text = []

            for sample in self.tokenized_text:
                without_symbol = [word for word in sample if word not in symbols and len(word) > 1]
                without_symbols_text.append(without_symbol)

            self.preprocessed_text = without_symbols_text
        self.avg_len = np.mean([len(text) for text in self.preprocessed_text])
        self.stdv_len = np.std([len(text) for text in self.preprocessed_text])
        self.max_len=np.max([len(text) for text in self.preprocessed_text])

    @classmethod
    def preprocess(cls, texts, stopwords_file):

        tokenized_text = [tokenize(text) for text in texts]

        preprocessed_text = preprocess(tokenized_text, stopwords_file)

        return preprocessed_text

    def get_features(self, preprocessed, max_seq_length):
        word_embedding = {}
        if not "odia" in self.dataset_file:
            with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word_embedding[values[0]] = np.asarray(values[1:], "float32")

            with open("unk.vec", "r", encoding="utf8") as unk:
                for line in unk:
                    uni_vector = line.split()
                    word_embedding[uni_vector[0]] = np.asarray(values[1:], "float32")
                    # print(word_embedding[uni_vector[0]])
        else:
            with open("fasttext.wiki.300d.vec", 'r', encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word_embedding[values[0]] = np.asarray(values[1:], "float32")

            with open("unk-odia.vec", "r", encoding="utf8") as unk:
                for line in unk:
                    uni_vector = line.split()
                    word_embedding[uni_vector[0]] = np.asarray(values[1:], "float32")
        print("Start getting the embeddings...")
        embeddings = []
        for row in preprocessed:
            row_embedding_vec = self.get_embedding_vector(word_embedding, row, max_seq_length)
            embeddings.append(row_embedding_vec)
        print("embedd",np.shape(embeddings))
        ##normalizing the data
        embeddings=[normalize(vector) for vector in embeddings]
        print("embedd after", np.shape(embeddings))
        return embeddings

    def get_embedding_vector(self, word_embedding, x, max_seq_length):
        """
        returns an embedding vector of size 50*max_seq_length or 300*max_seq_length. for the words not seen befores
        uses the average of all the vectors, for padding uses vectors of zeros
        :param word_embedding: dictionary mapping each word to a pretrianed vector
        :param x: ine sample of a data (sentence)
        :return:
        """
        embed_vec_len = len(list(word_embedding.values())[0])
        embedding_vector = np.zeros((max_seq_length,embed_vec_len))

        ### if the length of a sequence is less than the max seq length padd the vector with
        ### array of zeros
        if max_seq_length > len(x):

            for i, word in enumerate(x):
                if word not in word_embedding.keys():
                    embedding_vector[i] = word_embedding["UNK"]
                else:
                    embedding_vector[i] = word_embedding[word]
            for i in range(len(x), max_seq_length):
                embedding_vector[i] = np.zeros(embed_vec_len)
            # print("here",embedding_vector)
        else:
            for i, word in enumerate(x[:max_seq_length]):
                if word not in word_embedding.keys():
                    embedding_vector[i] = word_embedding["UNK"]
                else:
                    embedding_vector[i] = word_embedding[word]
        return embedding_vector

    def save_features(self, vectors, save_path):
        training_data = [vectors, self.numeric_classes]
        with open(save_path, "wb") as f:
            pkl.dump(training_data, f)

    def get_train_test_data(self, data, test_size, train_path, test_path):

        X_train, X_test, y_train, y_test = train_test_split(data, self.numeric_classes, test_size=test_size,
                                                            random_state=42
                                                            , stratify=self.numeric_classes)

        with open(train_path, "wb") as f:
            pkl.dump([X_train, y_train], f)
        with open(test_path, "wb") as d:
            pkl.dump([X_test, y_test], d)


if __name__ == '__main__':
    f1 = Features("datasets/products.train.txt", stopwords_file="stopwords.txt")
    print(f1.avg_len, f1.stdv_len)
    max_seq_length = int(f1.avg_len + f1.stdv_len)
    # max_seq_length=f1.max_len
    print(max_seq_length)

    X = f1.get_features(preprocessed=f1.preprocessed_text, max_seq_length=max_seq_length)
    print(len(X))
    f1.save_features(X,"produts_embeddigs.pkl")
    # f1.get_train_test_data(data=X, test_size=0.2, train_path="odiya_train.pkl", test_path="odiya_test.pkl")

"""

for questions and odiya  dataset i put the max len casue they were short
and for 4 dim i put lest than the average

"""