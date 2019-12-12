from model import Model
m = Model()
m.fit()
m.score()
m.serialize_model("model.pkl")