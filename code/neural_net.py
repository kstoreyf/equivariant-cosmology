import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

from fit import Fitter


#class NeuralNetManual(nn.Module):
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size=32, output_size=1):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size

        self.lin1 = nn.Linear(self.input_size, self.hidden_size)
        self.act1 = nn.SELU()
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)

        self.lin2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act2 = nn.SELU()
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)

        self.lin3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act3 = nn.SELU()
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)

        self.linfinal = nn.Linear(self.hidden_size, output_size)

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        nn.init.xavier_uniform_(self.lin3.weight)
        nn.init.zeros_(self.lin3.bias)
        nn.init.xavier_uniform_(self.linfinal.weight)
        nn.init.zeros_(self.linfinal.bias)
        self.double()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.bn1(x)

        x = self.lin2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.bn2(x)

        x = self.lin3(x)
        x = self.act3(x)
        x = self.dropout3(x)
        x = self.bn3(x)

        output = self.linfinal(x)
        return output

# via Derek Lim,
# https://github.com/cptq/SignNet-BasisNet/blob/main/GraphPrediction/layers/mlp.py
class NeuralNetList(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=6,
                 use_bn=True, use_ln=False, dropout=0.5, activation_name='selu',
                 residual=False):
        super(NeuralNetList, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.activation_name = activation_name

        activation_dict = {'relu': nn.ReLU(),
                           'selu': nn.SELU(),
                           'elu': nn.ELU()}
        if activation_name not in activation_dict:
            raise ValueError(f"Activation {activation_name} not recognized!")
        self.activation = activation_dict[activation_name]

        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(input_size, output_size))
        else:
            self.lins.append(nn.Linear(input_size, hidden_size))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_size))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_size))
            for layer in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_size, hidden_size))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_size))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_size))
            self.lins.append(nn.Linear(hidden_size, output_size))

        # initialization
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)            

        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

        self.double()


    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2,1)).transpose(2,1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x



class NNFitter(Fitter):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def set_up_training_data(self):
        self.A_train = self.construct_feature_matrix(self.x_train, 
                                        y_current=self.y_current_train,
                                        x_extra=self.x_extra_train,
                                        include_ones_feature=False
                                        )
        self.scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=0)
        #self.scaler = MinMaxScaler()
        #self.scaler = StandardScaler() 

        self.scaler.fit(self.A_train)
        A_train_scaled = self.scaler.transform(self.A_train)

        self.dataset_train = DataSet(A_train_scaled, self.y_train, 
                                y_var=self.y_uncertainties_train**2)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                          batch_size=128, shuffle=True,
                                          worker_init_fn=seed_worker,
                                          num_workers=0)

    def set_up_validation_data(self):
        self.A_valid = self.construct_feature_matrix(self.x_valid, 
                                        y_current=self.y_current_valid,
                                        x_extra=self.x_extra_valid,
                                        include_ones_feature=False
                                        )
        A_valid_scaled = self.scaler.transform(self.A_valid)

        self.dataset_valid = DataSet(A_valid_scaled, self.y_valid, 
                                y_var=self.y_uncertainties_valid**2)
        self.data_loader_valid = DataLoader(self.dataset_valid, 
                                          batch_size=32, shuffle=True,
                                          worker_init_fn=seed_worker,
                                          num_workers=0)

    def train_one_epoch(self, epoch_index):
        running_loss_train = 0.
        running_loss_valid = 0.
        losses_train = []
        for i, data in enumerate(self.data_loader_train):
            x, y, y_var = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            y_pred = self.model(x.double())
            # Compute the loss and its gradients
            #loss = self.criterion(y_pred.squeeze(), y, y_var)
            # squeeze all in case they are 1-dimc
            loss = self.criterion(y_pred.squeeze(), y.squeeze(), y_var.squeeze())
            loss.backward()

            # print(y[0])
            # print(y_pred[0])
            # print(y_var[0])

            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss_train += loss.item()
            losses_train.append(loss.item())
            # print("Epoch", epoch_index)
            # print("train")
            # print(y[0])
            # print(y_pred[0])
            # print(y_var[0])
            # print(loss.item())

        self.model.eval()
        for i, data_val in enumerate(self.data_loader_valid):
            x, y, y_var = data_val
            y_pred = self.model(x.double())
            loss = self.criterion(y_pred.squeeze(), y.squeeze(), y_var.squeeze())
            running_loss_valid += loss.item()
            # print("valid")
            # print(y[0])
            # print(y_pred[0])
            # print(y_var[0])
            # print(loss.item())

        #print(np.mean(losses_train), np.min(losses_train), np.max(losses_train))
        last_loss_train = running_loss_train / len(self.data_loader_train)
        last_loss_valid = running_loss_valid / len(self.data_loader_valid)
        print(f"Training epoch {epoch_index}, training loss {last_loss_train:.2f}, validation loss {last_loss_valid:.2f}")
        return last_loss_train, last_loss_valid



    def train(self, max_epochs=100, learning_rate=0.0001, fn_model=None, save_at_min_loss=True, patience=50):
        
        self.model = NeuralNetList(self.input_size, hidden_size=self.hidden_size, output_size=self.output_size)

        #self.criterion = nn.MSELoss()
        self.criterion = nn.GaussianNLLLoss()
        # weight decay = 0.01
        # scheduled lr - cos decay, warmup
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate,
                                          weight_decay=1e-2)
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                    T_max=32, # Maximum number of iterations.
                                    eta_min=1e-6) # Minimum learning rate.

        # Training loop
        self.loss_train = []
        self.loss_valid = []
        self.model.train()
        loss_valid_min = np.inf
        epoch_best = None
        state_dict_best = None
        early_stopper = EarlyStopper(patience=patience, min_delta=0)
        for epoch_index in range(max_epochs):
            last_loss_train, last_loss_valid = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and last_loss_valid < loss_valid_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_valid_min = last_loss_valid
            self.loss_train.append(last_loss_train)
            self.loss_valid.append(last_loss_valid)
            self.scheduler.step()
            if early_stopper.early_stop(last_loss_valid):   
                print("Stopping early because patience criterion hit")          
                break
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
        if fn_model is not None:
            # if save_at_min_loss=False, will just save the last epoch 
            self.save_model(fn_model, epoch=epoch_best)


    def predict(self, x, y_current=None, x_extra=None):

        A = self.construct_feature_matrix(x, y_current=y_current, x_extra=x_extra,
                                          include_ones_feature=False)

        A_scaled = self.scaler.transform(A)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(A_scaled).double())
        return y_pred.squeeze().numpy()


    def save_model(self, fn_model, epoch=None):
        if epoch is None:
            epoch = len(self.loss_valid)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'num_layers': self.model.num_layers,
                    'use_bn': self.model.use_bn,
                    'use_ln': self.model.use_ln,
                    'dropout': self.model.dropout,
                    'activation_name': self.model.activation_name,
                    'residual': self.model.residual,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler': self.scaler,
                    'loss_train': self.loss_train,
                    'loss_valid': self.loss_valid,
                    'epoch': epoch
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        model_checkpoint = torch.load(fn_model)
        if 'output_size' in model_checkpoint:
            output = model_checkpoint['output_size']
        else:
            # for back-compatibility
            output = 1

        self.model = NeuralNetList(model_checkpoint['input_size'], 
                               hidden_size=model_checkpoint['hidden_size'],
                               output_size=output,
                               num_layers=model_checkpoint['num_layers'],
                               use_bn=model_checkpoint['use_bn'],
                               use_ln=model_checkpoint['use_ln'],
                               dropout=model_checkpoint['dropout'],
                               activation_name=model_checkpoint['activation_name'],
                               residual=model_checkpoint['residual'],
                               )
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler = model_checkpoint['scaler']
        if 'loss_train' in model_checkpoint:
            self.loss_train = model_checkpoint['loss_train']
        if 'loss_valid' in model_checkpoint:
            self.loss_valid = model_checkpoint['loss_valid']
        if 'loss' in model_checkpoint:
            self.loss = model_checkpoint['loss']        
        self.epoch = model_checkpoint['epoch']



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    print('worker seed', worker_seed)


class DataSet(Dataset):

    def __init__(self, X, Y, y_var=None, randomize=True):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.y_var = y_var
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")
        if y_var is not None:
            self.y_var = np.array(self.y_var)
            if len(self.X) != len(self.y_var):
                raise Exception("The length of X does not match the length of y_var")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        if self.y_var is not None:
            _y_var = self.y_var[index]
            return _x, _y, _y_var
        return _x, _y


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False