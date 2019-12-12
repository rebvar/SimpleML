from .train import load_data_X_y, get_train_test_split, train_and_return_model, score_model
import joblib

class Model:
    
    def __init__(self):
        self.X, self.y = load_data_X_y()
        self.X_train, self.X_test, self.y_train, self.y_test = get_train_test_split (self.X, self.y)
    
    
    def fit(self):
        self.lm = train_and_return_model(self.X_train, self.y_train)
    
    
    
    def score(self):
        self._score = score_model(self.lm, self.X_test, self.y_test)
    
    def get_score(self):
        return self._score
    
    def serialize_model(self, model_name):
        with open(model_name, "wb") as file:
            joblib.dump(value=self.lm, filename=model_name)
        
        