import os

from tensorflow.keras.models import model_from_json


# define class
def read_data(path, data=[]):
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding="utf8") as file:
            data.append(file.read())

    return data


def save_nn_model(model, path, name):
    # Serialize to JSON
    json_file = model.to_json()
    file_name = path + "/" + name
    json_file = file_name + ".json"
    with open(json_file, "w") as file:
        file.write(json_file)

    # Serialize weights to HDF5
    model.save_weights(file_name + "_weights.h5")
    print("Model saved...")


# Method for loading saved model
def load_nn_model(filename):
    # Load JSON and create model

    json_file = filename + ".json"

    file = open(json_file, "r")
    model_json = file.read()
    file.close()

    loaded_model = model_from_json(model_json)
    # Load weights
    loaded_model.load_weights(filename + "_weights.h5")
    print("Model loaded successfully...")

    return loaded_model


def save_ml_model(model, path, name):
    import pickle
    pickle_out = open(path + "/" + name + ".pkl", "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()


def load_ml_model(path, name):
    import pickle
    pickle_in = open(path + "/" + name + ".pkl", "rb")
    model = pickle.load(pickle_in)

    return model
