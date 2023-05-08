import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor

from pytorch_tabnet.tab_model import TabNetRegressor

import xgboost as xgb
from xgboost import XGBRegressor 

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
        #self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=0)
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
        # self.model = HistGradientBoostingRegressor(max_iter=max_epochs,
        #                                            learning_rate=learning_rate)
        #self.model.fit(self.A_train_scaled, self.y_train.ravel())
        # need ravel bc targets are shape (N,1), and need (N,)
        y_variance_train = self.y_uncertainties_train**2

        self.models = []
        for i in range(self.y_train.shape[1]):
            model = HistGradientBoostingRegressor(max_iter=max_epochs,
                                                   learning_rate=learning_rate)
                                                   
            #y_variance_train = self.y_uncertainties_train[:,i]
            model.fit(self.A_train_scaled, self.y_train[:,i], sample_weight=1.0/y_variance_train[:,i])
            #model.fit(self.A_train_scaled, self.y_train[:,i])
            self.models.append(model)
        # booster = HistGradientBoostingRegressor(max_iter=max_epochs,
        #                                         learning_rate=learning_rate)
        # self.model = MultiOutputRegressor(booster)
        # self.model.fit(self.A_train_scaled, self.y_train)
        # self.model = XGBRegressor(n_estimators=100, learning_rate=learning_rate, max_depth=10)
        # y_train = np.array(list(self.y_train))
        #self.model.fit(np.array(self.A_train_scaled), y_train)
        #if fn_model is not None:
        #    self.save_model(fn_model)


    def predict(self, x, y_current=None, x_extra=None):
        A = self.construct_feature_matrix(x, y_current=y_current, x_extra=x_extra,
                                          include_ones_feature=False)
        A_scaled = self.scaler.transform(A)
        y_pred = np.empty((len(x), self.y_train.shape[1]))
        for i, model in enumerate(self.models):
            y_pred[:,i] = model.predict(A_scaled)
        return y_pred


    def save_model(self, fn_model, epoch=None):
        pass

    def load_model(self, fn_model):
        pass



class XGBoosterFitter(Fitter):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def set_up_training_data(self):
        self.A_train = self.construct_feature_matrix(self.x_train, 
                                        y_current=self.y_current_train,
                                        x_extra=self.x_extra_train,
                                        include_ones_feature=False
                                        )
        #self.scaler = MinMaxScaler() # TODO revisit !! 
        #self.scaler = StandardScaler() # TODO revisit !! 
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=0)
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

        #self.model = XGBRegressor(n_estimators=300, learning_rate=learning_rate, max_depth=10)
        self.model = XGBRegressor(n_estimators=100, learning_rate=learning_rate)
        y_train = np.array(list(self.y_train))
        self.model.fit(np.array(self.A_train_scaled), y_train)
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
        #self.scaler = StandardScaler() # TODO revisit !! 
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=0)
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
        #self.scaler = StandardScaler() # TODO revisit !! 
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=0)
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
    def train(self, max_epochs=500, learning_rate=0.02, fn_model=None, save_at_min_loss=True, loss_fn=None):

        optimizer_params = dict(lr=learning_rate)
        self.model = TabNetRegressor(optimizer_params=optimizer_params)
        # could include eval_set=[(X_valid, y_valid) for early stopping
        print("A:", self.A_train_scaled.shape)
        print("y:", self.y_train.shape)
        #self.model.fit(self.A_train_scaled, self.y_train, max_epochs=max_epochs)
        # if loss_fn is None:
        #     loss_fn = nn.MSELoss()

        self.model.fit(
                self.A_train_scaled, self.y_train,
                eval_set=[(self.A_train_scaled, self.y_train), 
                          (self.A_valid_scaled, self.y_valid)],
                eval_name=['train', 'valid'],
                eval_metric=['mae', 'rmse', 'mse'],
                max_epochs=max_epochs,
                patience=50, 
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
                #augmentations=aug,
                #loss_fn=loss_fn,
        )

        #if fn_model is not None:
        #    self.save_model(fn_model)


    def predict(self, x, y_current=None, x_extra=None):
        A = self.construct_feature_matrix(x, y_current=y_current, x_extra=x_extra,
                                          include_ones_feature=False)
        A_scaled = self.scaler.transform(A)
        y_pred = self.model.predict(A_scaled)

        # had this for a reason, can't for multi-d, may need to revisit
        #y_pred = np.array(y_pred).flatten()
        y_pred = np.array(y_pred)
        print("y_pred:", y_pred.shape)
        return y_pred


    def save_model(self, fn_model, epoch=None):
        pass

    def load_model(self, fn_model):
        pass
