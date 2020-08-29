from models.ml_model_template import MLModelTemplate


class LogisticModel(MLModelTemplate):

    def __init__(self):
        pass

    def create_model(self):
        from sklearn.linear_model.logistic import LogisticRegression
        classifier = LogisticRegression(max_iter=10000)

        return classifier


if __name__ == "__main__":
    logistic_model = LogisticModel()
    logistic_model.execute()
