import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

from fit import Fitter


class NeuralNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size=32, output_size=1):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.lin1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.act1 = torch.nn.SELU()
        self.lin2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act2 = torch.nn.SELU()
        self.lin3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act3 = torch.nn.SELU()
        self.linfinal = torch.nn.Linear(self.hidden_size, output_size)

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




class NNFitter(Fitter):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def set_up_training_data(self):
        self.A_train = self.construct_feature_matrix(self.x_train, 
                                        y_current=self.y_current_train,
                                        x_extra=self.x_extra_train,
                                        include_ones_feature=False
                                        )

        #self.scaler = StandardScaler(with_mean=True, with_std=False)
        self.scaler = MinMaxScaler() # TODO revisit !! 
        #self.scaler = QuantileTransformer(n_quantiles=10)
        self.scaler.fit(self.A_train)
        A_train_scaled = self.scaler.transform(self.A_train)

        self.dataset_train = DataSet(A_train_scaled, self.y_train, 
                                y_var=self.y_uncertainties_train**2)
        # g = torch.Generator()
        # g.manual_seed(0)
        self.data_loader_train = DataLoader(self.dataset_train, 
                                          batch_size=32, shuffle=True,
                                          worker_init_fn=seed_worker,
                                          #generator=g, 
                                          num_workers=0)

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(self.data_loader_train):
            x, y, y_var = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            y_pred = self.model(x.double())

            # Compute the loss and its gradients
            loss = self.criterion(y_pred.squeeze(), y, y_var)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        last_loss = running_loss / len(self.data_loader_train)
        print("Training epoch", epoch_index, 'loss', last_loss)
        return last_loss



    def train(self, max_epochs=100, learning_rate=0.0001, fn_model=None, save_at_min_loss=True):
        
        #self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.GaussianNLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.loss = []
        self.model.train()
        loss_min = np.inf
        epoch_best = None
        state_dict_best = None
        for epoch_index in range(max_epochs):
            last_loss = self.train_one_epoch(epoch_index)
            #print(last_loss, loss_min)
            if save_at_min_loss and fn_model is not None and last_loss < loss_min:
                #print(last_loss, loss_min)
                state_dict_best = self.model.state_dict()
                #print(state_dict_best)
                epoch_best = epoch_index
                loss_min = last_loss
            self.loss.append(last_loss)
        
        print('Epoch best:', epoch_best)
        # revert to state dict for model with lowest loss
        if save_at_min_loss:
            self.model.load_state_dict(state_dict_best)
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
            epoch = len(self.loss)
        save_dict = {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'output_size': self.model.output_size,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler': self.scaler,
                    'loss': self.loss,
                    'epoch': self.loss
                    }
        torch.save(save_dict, fn_model)


    def load_model(self, fn_model):
        print("hi!")
        model_checkpoint = torch.load(fn_model)
        print(model_checkpoint['output_size'])
        self.model = NeuralNet(model_checkpoint['input_size'], hidden_size=model_checkpoint['hidden_size'],
                               output_size=model_checkpoint['output_size'])
        self.model.load_state_dict(model_checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler = model_checkpoint['scaler']
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

