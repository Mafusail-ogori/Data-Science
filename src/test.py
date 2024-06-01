import pandas as pd 
import pickle 
from preprocessing import preprocess_data
from sklearn.metrics import classification_report
import torch

def test_model():
    ds = pd.read_csv('C:\\Users\\dpopr\\ITSS\\data\\new_input.csv')
    ds = preprocess_data(ds)

    X = ds.drop(['Status'], axis=1)
    y = ds['Status']    
    
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    
    y_tensor = y_tensor[:X_tensor.size(0)]
    
    with open('C:\\Users\\dpopr\\ITSS\\models\\mlp.pkl', 'rb') as f:
        model = pickle.load(f)
    
    y_pred = model.predict(X_tensor)
    report = classification_report(y, y_pred)
    print(report)
    
    
    pd.DataFrame(y_pred).to_csv('C:\\Users\\dpopr\\ITSS\\data\\predictions.csv')
