import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

from fit import Fitter




class BoosterFitter(Fitter):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def set_up_training_data(self):
        self.A_train = self.construct_feature_matrix(self.x_train, 
                                        y_current=self.y_current_train,
                                        x_extra=self.x_extra_train,
                                        include_ones_feature=False
                                        )
        #self.scaler = MinMaxScaler() # TODO revisit !! 
        self.scaler = StandardScaler() # TODO revisit !! 
        self.scaler.fit(self.A_train)
        self.A_train_scaled = self.scaler.transform(self.A_train)


    def set_up_validation_data(self):
        self.A_valid = self.construct_feature_matrix(self.x_valid, 
                                        y_current=self.y_current_valid,
                                        x_extra=self.x_extra_valid,
                                        include_ones_feature=False
                                        )
        self.A_valid_scaled = self.scaler.transform(self.A_valid)


    def train(self, max_epochs=100, learning_rate=0.1, fn_model=None, save_at_min_loss=True):
        print("lr:", learning_rate)
        # self.model = GradientBoostingRegressor(learning_rate=learning_rate,
        #                                       n_estimators=500)
        self.model = HistGradientBoostingRegressor(max_iter=max_epochs,
                                                   learning_rate=learning_rate)
        self.model.fit(self.A_train_scaled, self.y_train.ravel())

        #if fn_model is not None:
        #    self.save_model(fn_model)


    def predict(self, x, y_current=None, x_extra=None):
        A = self.construct_feature_matrix(x, y_current=y_current, x_extra=x_extra,
                                          include_ones_feature=False)
        A_scaled = self.scaler.transform(A)
        y_pred = self.model.predict(A_scaled)
        return y_pred


    def save_model(self, fn_model, epoch=None):
        pass

    def load_model(self, fn_model):
        pass








class RFFitter(Fitter):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def set_up_training_data(self):
        self.A_train = self.construct_feature_matrix(self.x_train, 
                                        y_current=self.y_current_train,
                                        x_extra=self.x_extra_train,
                                        include_ones_feature=False
                                        )
        #self.scaler = MinMaxScaler() # TODO revisit !! 
        self.scaler = StandardScaler() # TODO revisit !! 
        self.scaler.fit(self.A_train)
        self.A_train_scaled = self.scaler.transform(self.A_train)


    def set_up_validation_data(self):
        self.A_valid = self.construct_feature_matrix(self.x_valid, 
                                        y_current=self.y_current_valid,
                                        x_extra=self.x_extra_valid,
                                        include_ones_feature=False
                                        )
        self.A_valid_scaled = self.scaler.transform(self.A_valid)

    # keeping function signature the same
    def train(self, max_epochs=100, learning_rate=0.1, fn_model=None, save_at_min_loss=True):
        
        self.model = RandomForestRegressor(n_estimators=300)
        self.model.fit(self.A_train_scaled, self.y_train)

        #if fn_model is not None:
        #    self.save_model(fn_model)


    def predict(self, x, y_current=None, x_extra=None):
        A = self.construct_feature_matrix(x, y_current=y_current, x_extra=x_extra,
                                          include_ones_feature=False)
        A_scaled = self.scaler.transform(A)
        y_pred = self.model.predict(A_scaled)
        return y_pred


    def save_model(self, fn_model, epoch=None):
        pass

    def load_model(self, fn_model):
        pass


class TabNetFitter(Fitter):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def set_up_training_data(self):
        self.A_train = self.construct_feature_matrix(self.x_train, 
                                        y_current=self.y_current_train,
                                        x_extra=self.x_extra_train,
                                        include_ones_feature=False
                                        )
        #self.scaler = MinMaxScaler() # TODO revisit !! 
        self.scaler = StandardScaler() # TODO revisit !! 
        self.scaler.fit(self.A_train)
        self.A_train_scaled = self.scaler.transform(self.A_train)


    def set_up_validation_data(self):
        self.A_valid = self.construct_feature_matrix(self.x_valid, 
                                        y_current=self.y_current_valid,
                                        x_extra=self.x_extra_valid,
                                        include_ones_feature=False
                                        )
        self.A_valid_scaled = self.scaler.transform(self.A_valid)

    # keeping function signature the same
    def train(self, max_epochs=100, learning_rate=0.02, fn_model=None, save_at_min_loss=True):

        optimizer_params = dict(lr=learning_rate)
        self.model = TabNetRegressor(optimizer_params=optimizer_params)
        # could include eval_set=[(X_valid, y_valid) for early stopping
        self.model.fit(self.A_train_scaled, self.y_train, max_epochs=max_epochs)

        #if fn_model is not None:
        #    self.save_model(fn_model)


    def predict(self, x, y_current=None, x_extra=None):
        A = self.construct_feature_matrix(x, y_current=y_current, x_extra=x_extra,
                                          include_ones_feature=False)
        A_scaled = self.scaler.transform(A)
        y_pred = self.model.predict(A_scaled)
        y_pred = np.array(y_pred).flatten()
        return y_pred


    def save_model(self, fn_model, epoch=None):
        pass

    def load_model(self, fn_model):
        pass