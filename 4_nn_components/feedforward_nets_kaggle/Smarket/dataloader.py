import pandas as pd

class DatasetHandler:
    def __init__(self):
        pass
    
    @staticmethod
    def create_train_test(data: pd.DataFrame, y_col:str, sample_fraction:int=.8):
        # create train_x, train_y, test_x, test_y.
        train = data.sample(frac=sample_fraction, random_state=42)
        train_x = train.drop(columns=[y_col])
        train_y = train[y_col]

        test = data.drop(train.index)
        test_x = test.drop(columns=[y_col])
        test_y = test[y_col]

        assert train_x.shape[0] == train_y.shape[0]
        assert test_x.shape[0] == test_y.shape[0]
        print(f"Number of train record={train_x.shape}, test records={test_x.shape}")

        return (train_x, train_y, test_x, test_y)
