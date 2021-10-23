from abc import ABCMeta, abstractmethod
import pickle
from sklearn.metrics import  f1_score,mean_squared_error,classification_report

class Model(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    def save_model(self, model):
        with open(self.model_file, "wb") as file:
            pickle.dump(model, file)

    def load_model(self):
        with open(self.model_file, "rb") as file:
            model = pickle.load(file)
        return model

    def get_accuracy(self,y_pred,y_true):
        tp_tn=len([label for i,label in enumerate(y_pred) if label==y_true[i]])
        acc=tp_tn/len(y_pred)
        return acc
    def train(self, input_file):
        pass

    @abstractmethod
    def classify(self, input_file, model):
        pass
