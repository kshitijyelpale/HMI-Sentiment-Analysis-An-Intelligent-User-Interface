from ml_model_template import MLModelTemplate


class DecisionTreeModel(MLModelTemplate):

    def __init__(self):
        pass

    def create_model(self):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(random_state=0)

        return classifier


if __name__ == "__main__":
    decision_tree_model = DecisionTreeModel()
    decision_tree_model.execute()
