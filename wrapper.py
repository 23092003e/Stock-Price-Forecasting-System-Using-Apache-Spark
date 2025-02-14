#!/usr/bin/env python
# coding: utf-8

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler, MinMaxScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import torch

class RegressorWrapper:
    def __init__(self, model, evaluator, target, num=None, cat=None):
        self.model = model
        self.ev = evaluator.setLabelCol(target)
        self.target = target
        self.num = num
        self.cat = cat

    def __preprocess(self, data) -> Pipeline:
        # duplicates
        data.drop_duplicates()

        # pipeline stages
        stages = []
        features = []
        if self.num:
            num_assembler = VectorAssembler(inputCols=self.num,
                                            outputCol='num_features')
            scaler = StandardScaler(inputCol='num_features',
                                    outputCol='scaled_features')
            stages += [num_assembler, scaler]
            features.append('scaled_features')
            
        if self.cat:
            indexer = [StringIndexer(inputCol=c,
                                     outputCol=f'idx_{c}',
                                     handleInvalid='keep') for c in self.cat]
            encoder = [OneHotEncoder(inputCol=f'idx_{c}',
                                     outputCol=f'ohe_{c}') for c in self.cat]
            stages += indexer + encoder
            features += [f'ohe_{c}' for c in self.cat]

        assembler = [VectorAssembler(inputCols=features, outputCol='features')]
        stages += assembler

        # add model to stages
        self.model.setFeaturesCol('features')
        self.model.setLabelCol(self.target)
        stages.append(self.model)

        # return a pipeline
        return Pipeline(stages = stages)

    def __cross_validate(self, data, paramGrid, folds):
        # create pipeline
        pipeline = self.__preprocess(data)

        # initialize cross-validate
        cv = CrossValidator(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=self.ev, numFolds=folds)

        return cv.fit(data)

    def create(self, data , paramGrid, metrics, folds=5, seed=42):
        # split data for evaluation
        train, test = data.randomSplit([0.8, 0.2], seed=seed)

        # get result
        cv_model = self.__cross_validate(train, paramGrid, folds)
        pred_train = cv_model.transform(train)

        best_model = cv_model.bestModel
        pred_test = best_model.transform(test)

        train_metrics = {}
        test_metrics = {}
        for m in metrics:
            train_metrics[m] = self.ev.setMetricName(m).evaluate(pred_train)
            test_metrics[m] = self.ev.setMetricName(m).evaluate(pred_test)

        return {'train':train_metrics, 'test': test_metrics, 'model': best_model}

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, X):
        # LSTM 1-3
        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            X, (hidden, _) = lstm(X)
            
        # fully connected layers
        X = self.relu(self.fc1(hidden[-1]))
        X = self.fc2(X)
        return X
        
class LSTMWrapper:
    def __preprocess(self, data):
        # attributes scaling
        assembler = VectorAssembler(inputCols=data.columns, outputCol='vars')
        scaler = MinMaxScaler(min=0.0, max=1.0, inputCol='vars', outputCol='scaled_vars')
        pipeline = Pipeline(stages=[assembler, scaler])

        # train test split
        train, test = data.randomSplit([0.8, 0.2], seed=42)
        transformer = pipeline.fit(train)
        train = transformer.transform(train)
        test = transformer.transform(test)

        # return data as Numpy array
        return (train.select('scaled_vars').rdd.map(lambda row: row[0][0]).collect(),
                test.select('scaled_vars').rdd.map(lambda row: row[0][0]).collect()) 
    
    def __create_sequences(self, data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    def create(self, df, input_dim, output_dim,
               epoches=200, sequence_length=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # split train and test
        train, test = self.__preprocess(df)
        
        # get x and y
        X_train, y_train = self.__create_sequences(train, sequence_length)
        X_test, y_test = self.__create_sequences(test, sequence_length)

        # conver to pytorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

        # create dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # create model
        model = LSTMModel(input_dim=input_dim, output_dim=output_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # train model
        for epoch in range(epoches):
            # train loop
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs = inputs.unsqueeze(-1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.unsqueeze(-1)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                    val_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epoches}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}")