
import pickle
from dataclasses import dataclass
import logging
import pandas as pd 
import time


@dataclass
class PredictionEngine:
    file_name: str = './src/rsrcs/SVM092520231931_model.pkl'


    def __post_init__(self):
        self.model = self.load_model(self.file_name)


    def load_sample_from_df(self, df: pd.DataFrame):
        try:
            if len(df.columns) != 312:
                raise ValueError("Invalid number of columns in dataframe passed to load_sample_from_df")
            else:
                X = df.iloc[0:, 1:-1].astype(int)
                return X
        except Exception as e:
            logging.error(f"Error building json from df: {str(e)}")
            return None

    def batch_predict(self, df: pd.DataFrame):
        
        try:
            if len(df.columns) != 312:
                raise ValueError("Invalid number of columns in dataframe passed to load_sample_from_df")
            else:
                labels = []
                if self.model is not None:
                    for index in df.index:
                        sliced_df = df.loc[[index]]
                        
                        X = sliced_df.iloc[:, 1:-1].astype(int)
                        
                        label = self.classify_sample(X)
                        labels.append(label)
                        # df.loc[index, 'performance_label'] = label
                        # print(f'Row {index} classified as {label}')
                    
                return labels
            
        except Exception as e:
            print(e)
            logging.error(f"Error batch predicting from df: {str(e)}")
            return None

    def load_model(self, file_name):
        try:
            with open(file_name, 'rb') as file:
                model = pickle.load(file)
                if not hasattr(model, "predict"):
                    raise ValueError("Invalid pickle model")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {file_name}. Error: {str(e)}")
            return None


    def load_sample(self, json_file):
        try:
            data = pd.read_json(json_file)
            X = data.astype(int)
            return X        
        except Exception as e:
            logging.error(f"Error loading json from {json_file}. Error: {str(e)}")
            return None


    def classify_sample(self, sample):
        try:                
            return self.model.predict(sample)[0]
        except Exception as e:
            logging.error(f"Error classifying sample: {str(e)}")
            return None


    def predict(self, json_file: str = None, df: pd.DataFrame = None):
        
        if json_file is None and df is None:
            raise ValueError("Must provide either json_file or df")
        elif json_file is not None:
            X = self.load_sample(json_file)
        elif df is not None:
            X = self.load_sample_from_df(df)
            print(X)
        
        if self.model is not None:
            label = self.classify_sample(X)
            # print('sample classified as',label)
            return label
