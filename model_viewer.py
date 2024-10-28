import pickle

# Replace 'your_model.pkl' with the path to your .pkl file
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
