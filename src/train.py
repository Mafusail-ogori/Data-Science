import torch 
import pandas as pd
from preprocessing import preprocess_data
from sklearn.neural_network import MLPClassifier
import pickle

def train_model():
    ds = pd.read_csv('C:\\Users\\dpopr\\ITSS\\data\\train.csv')
    ds = preprocess_data(ds)

    X = ds.drop(['Status'], axis=1)
    y = ds['Status']    
    
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    
    y_tensor = y_tensor[:X_tensor.size(0)]
    mlp = MLPClassifier(alpha=0.0001, hidden_layer_sizes=64, learning_rate_init=0.001, solver='adam', activation='tanh', max_iter=200)
    mlp.fit(X_tensor, y_tensor)
    
    with open('C:\\Users\\dpopr\\ITSS\\models\\mlp.pkl', 'wb') as f:
        pickle.dump(mlp, f)
    