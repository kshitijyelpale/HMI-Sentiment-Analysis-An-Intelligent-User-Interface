
from models.Model_LSTM_1 import LSTMModel

print("Hello world")

import pathlib
print(pathlib.Path("__file__").parent.absolute())
import os

path = os.path.dirname(__file__)

os.chdir(path)

print(os.path.dirname(__file__))
