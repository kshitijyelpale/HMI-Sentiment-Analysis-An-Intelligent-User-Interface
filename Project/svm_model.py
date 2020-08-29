from ml_model_template import MLModelTemplate


class SimpleSVMModel(MLModelTemplate):

    def __init__(self):
        pass

    def create_model(self):
        from sklearn.svm import SVC
        classifier = SVC()

        return classifier


if __name__ == "__main__":
    svm_model = SimpleSVMModel()
    svm_model.execute()
