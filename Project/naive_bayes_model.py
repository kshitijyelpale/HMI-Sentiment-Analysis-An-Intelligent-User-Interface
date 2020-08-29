from ml_model_template import MLModelTemplate


class NaiveBayesModel(MLModelTemplate):

    def __init__(self):
        pass

    def create_model(self):
        # from sklearn.naive_bayes import GaussianNB
        # classifier = GaussianNB()

        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB()

        return classifier

    def predict_reviews(self, raw_data):
        temp = []
        temp, data = self.pre_process_data(temp, raw_data)

        # print(data)
        # from datapreprocessing import tfidfvectorizer
        # data = tfidfvectorizer.transform(data)
        vec = self.load_model("vec")
        data = vec.transform(data)
        # print(data)
        # print(type(data))
        model = self.load_model()

        output = self.predict(model, data)

        return output


if __name__ == "__main__":
    nb_model = NaiveBayesModel()
    # nb_model.execute()
    reviews = ["I am very happy after watching this movie", "i did not like it"]

    print(nb_model.predict_reviews(reviews))
