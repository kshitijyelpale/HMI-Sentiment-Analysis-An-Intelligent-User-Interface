from models.ml_model_template import MLModelTemplate


class NaiveBayesModel(MLModelTemplate):

    def __init__(self):
        pass

    def create_model(self):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()

        # from sklearn.naive_bayes import MultinomialNB
        # classifier = MultinomialNB()

        return classifier


if __name__ == "__main__":
    nb_model = NaiveBayesModel()
    nb_model.execute()
