import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.simplefilter('ignore')
 
def impute_na_with_median(df):
    medians = []
    numerical_columns = df.select_dtypes(include=['float64', 'int64'])
    if numerical_columns.empty:
        print("No numerical columns found in the DataFrame.")
        return
    for col in numerical_columns.columns:
        medians.append(df[col].median())
    for col, median_value in zip(numerical_columns.columns, medians):
        df[col].fillna(median_value, inplace=True)

def encode_column(col_name, data): 
    encoder = OneHotEncoder(categories='auto', drop='first', handle_unknown='error')
    encoder.fit(data[[col_name]].fillna('Missing'))
    encoded_data = encoder.transform(data[[col_name]].fillna('Missing'))
    return encoded_data, encoder

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop([data.columns[0], 'ID', 'year'],axis = 1)

    impute_na_with_median(data)
 
    for col in data.select_dtypes(include=['object']): 
        col_name = data[col].name
        encoded_data, encoder = encode_column(col_name, data)
        data.drop(columns=[col_name], inplace=True)
        new_col_names = encoder.get_feature_names_out([col_name])
        encoded_df = pd.DataFrame(encoded_data.toarray(), columns=new_col_names)
        data = pd.concat([data, encoded_df], axis=1)
    print(' Modi Ds shape >> ', data.shape[1])    
    return data
