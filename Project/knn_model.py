from ml_model_template import MLModelTemplate


class KNNModel(MLModelTemplate):

    def __init__(self):
        pass

    def create_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5)

        return classifier


if __name__ == "__main__":
    knn_model = KNNModel()
    knn_model.execute()
