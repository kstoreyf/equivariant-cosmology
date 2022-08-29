import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fit import Fitter


class NeuralNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size=32):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act1 = torch.nn.SELU()
        self.lin2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act2 = torch.nn.SELU()
        self.lin3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act3 = torch.nn.SELU()
        self.linfinal = torch.nn.Linear(self.hidden_size, 1)

        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        torch.nn.init.zeros_(self.lin3.bias)
        torch.nn.init.xavier_uniform_(self.linfinal.weight)
        torch.nn.init.zeros_(self.linfinal.bias)
        self.double()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        output = self.linfinal(x)
        return output


def save_nn(nn, fn_nn):
    torch.save(nn.state_dict(), fn_nn)


def load():
    model = NeuralNet(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()


class NNFitter(Fitter):

    def __init__(self, x_scalar_features, y_scalar, 
                 y_val_current, x_features_extra=None, 
                 uncertainties=None):
        super().__init__(x_scalar_features, y_scalar, 
                 y_val_current, x_features_extra=x_features_extra,
                 uncertainties=uncertainties)


    def set_up_data(self, log_x=False, log_y=False):

        self.log_x, self.log_y = log_x, log_y
        self.scale_y_values()
        self.x_scalar_train_scaled = self.scale_x_features(self.x_scalar_train)
        if self.x_features_extra is None:
            self.x_features_extra_train_scaled = None
        else:
            self.x_features_extra_train_scaled = self.scale_x_features(self.x_features_extra_train)
        A_train = self.construct_feature_matrix(self.x_scalar_train_scaled, 
                                        self.y_val_current_train_scaled,
                                        x_features_extra=self.x_features_extra_train_scaled,
                                        training_mode=True)

        #self.scaler = StandardScaler()
        self.scaler = MinMaxScaler() # TODO revisit !! 
        self.scaler.fit(A_train)
        A_train_scaled = self.scaler.transform(A_train)
        self.dataset_train = DataSet(A_train_scaled, self.y_scalar_train_scaled, 
                                y_var=self.uncertainties_train_scaled**2)
        print("SHUFFLE FALSE")
        #self.data_loader_train = iter(DataLoader(self.dataset_train, batch_size=32, shuffle=False))


    def construct_feature_matrix(self, x_features, y_current, x_features_extra=None, training_mode=False):
        y_current = np.atleast_2d(y_current).T
        if x_features_extra is None:
            A = np.concatenate((y_current, x_features), axis=1)
        else:
            A = np.concatenate((y_current, x_features_extra, x_features), axis=1)           
        if training_mode:
            n_extra = x_features_extra.shape[1] if x_features_extra is not None else 0
            self.n_extra_features = 1 + n_extra # 2 for y_current
            self.n_A_features = self.n_x_features + self.n_extra_features
        return A


    def train(self, max_epochs=500, learning_rate=0.01):

        #self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.GaussianNLLLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.loss = []
        self.model.train()
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            # Forward pass
            # TODO: this y_var won't work if we don't pass unc to dataloader
            #x, y = self.data_loader_train.next() 
            x, y, y_var = self.data_loader_train.next() 
            if epoch==0:
                print(x[0][0], y[0])
                #print(x[0], y[0])
            y_pred = self.model(x.double())
            # Compute Loss
            #loss = self.criterion(y_pred.squeeze(), y)
            loss = self.criterion(y_pred.squeeze(), y, y_var)

            self.loss.append(loss.item())
            #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            # Backward pass
            loss.backward()
            optimizer.step()


    def predict(self, x, y_current, x_extra=None):
        x_scaled = self.scale_x_features(x)
        y_current_scaled = self.scale_y(y_current)
        A = self.construct_feature_matrix(x_scaled, y_current_scaled, x_features_extra=x_extra)

        A_scaled = self.scaler.transform(A)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.from_numpy(A_scaled).float())
        return y_pred.squeeze().numpy()


    def predict_test(self):
        print('xtest', self.x_scalar_test[0][0])
        self.x_scalar_test_scaled = self.scale_x_features(self.x_scalar_test)
        if self.x_features_extra_test is None:
            self.x_features_extra_test_scaled = None
        else:
            self.x_features_extra_test_scaled = self.scale_x_features(self.x_features_extra_test)
        self.A_test = self.construct_feature_matrix(self.x_scalar_test_scaled, 
                                                    self.y_val_current_test_scaled,
                                                    x_features_extra=self.x_features_extra_test_scaled)

        A_test_scaled = self.scaler.transform(self.A_test)
        self.model.eval()
        with torch.no_grad():
            #y_scalar_pred = self.model(torch.from_numpy(A_test_scaled).float())
            y_scalar_pred = self.model(torch.from_numpy(A_test_scaled).double())
            self.y_scalar_pred = y_scalar_pred.squeeze().numpy()
        print('ypred', self.y_scalar_pred[0])

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

