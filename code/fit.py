import numpy as np


class Fitter:

    def __init__(self, x_features=None, y=None, 
                 y_current=None, x_features_extra=None, uncertainties=None,
                 train_mode=True, predict_mode=False):

        if train_mode:
            msg = "Must pass x_features and y in train mode!"
            assert x_features is not None and y is not None, msg

            self.x_features = np.array(x_features)
            self.y = np.array(y)
            # y_current is our current best-guess for the y value, 
            # e.g. from a broken power law model of the stellar-to-halo mass relation
            self.x_features_extra = x_features_extra
            self.y_current = y_current 

            self.N_halos = x_features.shape[0]
            assert self.y.shape[0]==self.N_halos, "Must have same number of halos for x features and y labels!"
            
            if y_current is not None:
                self.y_current = np.array(y_current)
                assert self.y_current.shape[0]==self.N_halos, "Must have same number of halos for x features and y val current!"
            
            if x_features_extra is not None:
                self.x_features_extra = np.array(self.x_features_extra)
                assert self.x_features_extra.shape[0]==self.N_halos, "Must have same number of halos for x scalar features and extra features!"

            # TODO: not sure this makes sense as default?? either make required, or 
            # should probs keep checking if none
            if uncertainties is not None:
                self.uncertainties = np.array(uncertainties)
                assert self.uncertainties.shape[0]==self.N_halos, "Must have same number of halos for x features and uncertainties!"


    def scale_x_features(self, x_input):
        x = np.copy(x_input)
        if self.log_x:
           x = np.log10(x)
        return x


    def scale_y(self, y_input):
        y = np.copy(y_input)
        if self.log_y:
            y = np.log10(y)
        return y


    def scale_uncertainties(self, uncertainties_input, y_input):
        # will need to manually compute derivatives to figure out y uncertainty scaling!
        # reference: http://openbooks.library.umass.edu/p132-lab-manual/chapter/uncertainty-for-natural-logarithms/
        uncertainties = np.copy(uncertainties_input)
        if self.log_y:
            uncertainties /= y_input
        return uncertainties

    def unscale_y(self, y_input):
        y = np.copy(y_input)
        if self.log_y:
            y = 10**y
        return y


    def split_train_test(self, idx_train, idx_test):

        self.idx_train = idx_train
        self.idx_test = idx_test

        # Split train and test arrays
        self.x_scalar_train = self.x_features[self.idx_train]
        self.x_scalar_test = self.x_features[self.idx_test]
        self.y_train = self.y[self.idx_train]
        self.y_test = self.y[self.idx_test]

        # Split uncertainties, y_currents, extra features if exist
        if self.uncertainties is not None:
            self.uncertainties_train = self.uncertainties[self.idx_train]
            self.uncertainties_test = self.uncertainties[self.idx_test]

        if self.y_current is not None:
            self.y_current_train = self.y_current[self.idx_train]
            self.y_current_test = self.y_current[self.idx_test]

        if self.x_features_extra is None:
            self.x_features_extra_train = None
            self.x_features_extra_test = None
        else:
            self.x_features_extra_train = self.x_features_extra[self.idx_train]
            self.x_features_extra_test = self.x_features_extra[self.idx_test]

        # Set number values
        self.n_train = len(self.x_scalar_train)
        self.n_test = len(self.x_scalar_test)
        self.n_x_features = self.x_features.shape[1]

        if self.n_x_features > self.n_train/2:
            print('WARNING!!! Number of features ({self.n_features}) greater than half the number of training samples ({self.n_train})')


    def scale_y_values(self):
        self.y_train_scaled = self.scale_y(self.y_train)
        self.y_test_scaled = self.scale_y(self.y_test)
        self.uncertainties_train_scaled = self.scale_uncertainties(self.uncertainties_train, self.y_train)
        self.y_current_train_scaled = self.scale_y(self.y_current_train)
        self.y_current_test_scaled = self.scale_y(self.y_current_test)


    def construct_feature_matrix(self, x_features, y_current=None, 
                                 x_features_extra=None, include_ones_feature=True, training_mode=False):
                                 
        print("hi!")
        A = np.empty((x_features.shape[0], 0))
        if include_ones_feature:
            ones_feature = np.ones((x_features.shape[0], 1))
            A = np.hstack((A, ones_feature))
        if y_current is not None:
            y_current = np.atleast_2d(y_current).T
            A = np.hstack((A, y_current))
        if x_features_extra is not None:
            A = np.hstack((A, x_features_extra))
        A = np.hstack((A, x_features))   
        if training_mode:
            # TODO: do i need this?
            self.n_A_features = A.shape[1]
            self.n_extra_features = self.n_A_features - self.n_x_features
        return A


class LinearFitter(Fitter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def scale_and_fit(self, rms_x=False, log_x=False, log_y=False, check_cond_of_A=False, 
                            fit_mode='leastsq', regularization_lambda=0.0):
        self.log_x, self.log_y = log_x, log_y
        self.scale_y_values()
        self.x_scalar_train_scaled = self.scale_x_features(self.x_scalar_train)
        if self.x_features_extra is None:
            self.x_features_extra_train_scaled = None
        else:
            self.x_features_extra_train_scaled = self.scale_x_features(self.x_features_extra_train)
        self.A_train = self.construct_feature_matrix(self.x_scalar_train_scaled, self.y_current_train_scaled,
                                                     x_features_extra=self.x_features_extra_train_scaled,
                                                     training_mode=True)

        if rms_x:
            x_fitscales = np.sqrt(np.mean(self.A_train**2, axis=0))
            # avoid divide by zero
            x_fitscales[x_fitscales==0] = 1
        else:
            x_fitscales = np.ones(self.A_train.shape[1])

        # "scaled" denotes pre-done x-scalings to data, e.g. log
        # "fitscaled" denotes scaling just for the fit, and then quickly scaled out of the best-fit vector
        self.A_train_fitscaled = self.A_train / x_fitscales

        # in this code, A=x_vals, diag(C_inv)=inverse_variances, Y=y_vals
        if check_cond_of_A:
            u, s, v = np.linalg.svd(self.A_train, full_matrices=False)
            print('x_vals condition number:',  np.max(s)/np.min(s))
        inverse_variances = 1/self.uncertainties_train_scaled**2
        self.AtCinvA = self.A_train_fitscaled.T @ (inverse_variances[:,None] * self.A_train_fitscaled)
        self.AtCinvY = self.A_train_fitscaled.T @ (inverse_variances * self.y_train_scaled)
    
        # Regularize (if regularization_lambda=0, no regularization)
        lhs = self.AtCinvA + regularization_lambda * np.sum(inverse_variances) * np.identity(self.AtCinvA.shape[0])

        if fit_mode=='leastsq':
            res = np.linalg.lstsq(lhs, self.AtCinvY, rcond=None)
            self.theta_fitscaled = res[0]
            self.rank = res[2]
            s = res[3]
            self.condition_number = np.max(s)/np.min(s)
        elif fit_mode=='solve':
            self.theta_fitscaled = np.linalg.solve(lhs, self.AtCinvY)
            self.rank = np.linalg.matrix_rank(self.AtCinvA)
            u, s, v = np.linalg.svd(self.AtCinvA, full_matrices=False)
            self.condition_number = np.max(s)/np.min(s)
        else:
            raise ValueError(f"Input fit_mode={fit_mode} not recognized! Use one of: ['leastsq', 'solve']")
        self.theta = self.theta_fitscaled / x_fitscales

        # This chi^2 is in units of the data as given, so does not include the mass_multiplier
        self.y_train_pred = self.predict_from_A(self.A_train)
        self.chi2 = np.sum((self.y_train - self.y_train_pred)**2 * inverse_variances)

        assert len(self.theta) == self.A_train.shape[1], 'Number of coefficients from theta vector should equal number of features!'

    
    def predict_test(self):
        self.x_scalar_test_scaled = self.scale_x_features(self.x_scalar_test)
        if self.x_features_extra_test is None:
            self.x_features_extra_test_scaled = None
        else:
            self.x_features_extra_test_scaled = self.scale_x_features(self.x_features_extra_test)
        self.A_test = self.construct_feature_matrix(self.x_scalar_test_scaled, self.y_current_test_scaled,
                                                    x_features_extra=self.x_features_extra_test_scaled)
        self.y_pred = self.predict_from_A(self.A_test)


    def predict(self, x, y_current, x_extra=None):
        x_scaled = self.scale_x_features(x)
        y_current_scaled = self.scale_y(y_current)
        A = self.construct_feature_matrix(x_scaled, y_current_scaled, x_features_extra=x_extra)
        y_pred = self.predict_from_A(A)
        return y_pred


    def predict_from_A(self, A):
        # A is assumed to be NOT "fitscaled" by x_fitscales
        # but A is "scaled": pre-scaled by other means, e.g. log
        y_pred_scaled = A @ self.theta
        y_pred = self.unscale_y(y_pred_scaled)
        return y_pred


    