import os
import re
from ml_utilities import read_data

def import_data():
    print("Importing Dataset... Please wait!")

    # Import Training and Testing Data
    ''' y_train = [0] * 22750 + [1] * 22750
    y_test = [0] * 2250 + [1] * 2250 '''
    x_train, y_train=[], []
    x_test, y_test=[], []
    path = os.path.dirname(__file__)

    dataset_path = path + "/DataSet/"

    test_data_path = dataset_path + "Test/"
    pos_test_data_path = test_data_path + "Positive"
    neg_test_data_path = test_data_path + "Negative"

    train_data_path = dataset_path + "Train/"
    pos_train_data_path = train_data_path + "Positive"
    neg_train_data_path = train_data_path + "Negative"

    x_train, y_train = read_data(pos_train_data_path)
    x_train, y_train = read_data(neg_train_data_path, x_train, y_train)

    x_test, y_test = read_data(pos_test_data_path)
    x_test, y_test = read_data(neg_test_data_path, x_test, y_test)

    print("Dataset successfully imported...!!!")
    return x_train, y_train, x_test, y_test


# Method for pre-processing data
def remove_stopwords_and_special_chars(data):
    import nltk
    try:
        from nltk.corpus import stopwords
    except:
        nltk.download('stopwords')
        from nltk.corpus import stopwords

    # Fetch all stopwords and keep required stopwords
    all_stopwords = stopwords.words('english')

    stopwords_excluded = (
        "against", "up", "down", "out", "off", "over", "under", "more", "most", "each", "few", "some", "such", "no",
        "nor", "not", "only", "too", "very", "don", "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
        "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
        'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't")

    my_stopwords = [word for word in all_stopwords if word not in stopwords_excluded]

    # Get rid of special characters
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for i in range(0, len(data)):
        # Keep only alphabets with single whitespace
        data[i] = REPLACE_NO_SPACE.sub("", data[i].lower())
        data[i] = REPLACE_WITH_SPACE.sub(" ", data[i])

        # Remove unwanted stopwords
        data[i] = data[i].split()
        data[i] = [word for word in data[i] if word not in my_stopwords]
        data[i] = " ".join(data[i])

    return data


def onehot_encoding(data, max_features=50000, max_doc_length=100):
    from tensorflow.keras.preprocessing.text import one_hot
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    data = [one_hot(document, max_features) for document in data]

    # Add Bias
    for i in range(0, len(data)):
        data[i] = [1] + data[i]

    # Word Embedding
    data = pad_sequences(data, truncating='post', padding='post', maxlen=max_doc_length)

    return data


def tfidfvectorizer(data, test_data=[]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')

    data = vectorizer.fit_transform(data)
    if test_data:
        test_data = vectorizer.transform(test_data)
        return vectorizer, data, test_data

    return vectorizer, data
