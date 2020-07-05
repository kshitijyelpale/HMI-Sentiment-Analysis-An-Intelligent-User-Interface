from ml_model_template import MLModelTemplate


class RandomForestModel(MLModelTemplate):

    def __init__(self):
        pass

    def create_model(self):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(max_depth=2, random_state=0)

        return classifier


if __name__ == "__main__":
    random_forest_model = RandomForestModel()
    random_forest_model.execute()
