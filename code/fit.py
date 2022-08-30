import numpy as np


class Fitter:

    def __init__(self):
        pass


    def load_training_data(self, x_train, y_train, 
                           y_current_train=None, x_extra_train=None, 
                           y_uncertainties_train=None):

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        # y_current is our current best-guess for the y value, 
        # e.g. from a broken power law model of the stellar-to-halo mass relation
        self.x_extra_train = x_extra_train
        self.y_current_train = y_current_train
        self.y_uncertainties_train = y_uncertainties_train

        self.N_halos = x_train.shape[0]
        assert self.y_train.shape[0]==self.N_halos, "Must have same number of halos for x features and y labels!"
        
        if y_current_train is not None:
            self.y_current_train = np.array(y_current_train)
            assert self.y_current_train.shape[0]==self.N_halos, "Must have same number of halos for x features and y val current!"
        
        if x_extra_train is not None:
            self.x_extra_train = np.array(self.x_extra_train)
            assert self.x_extra_train.shape[0]==self.N_halos, "Must have same number of halos for x scalar features and extra features!"

        # TODO: not sure this makes sense as default?? either make required, or 
        # should probs keep checking if none
        if y_uncertainties_train is not None:
            self.y_uncertainties_train = np.array(y_uncertainties_train)
            assert self.y_uncertainties_train.shape[0]==self.N_halos, "Must have same number of halos for x features and y_uncertainties!"



    def construct_feature_matrix(self, x, y_current=None, 
                                 x_extra=None, include_ones_feature=True):
        A = np.empty((x.shape[0], 0))
        if include_ones_feature:
            ones_feature = np.ones((x.shape[0], 1))
            A = np.hstack((A, ones_feature))
        if y_current is not None:
            y_current = np.atleast_2d(y_current).T
            A = np.hstack((A, y_current))
        if x_extra is not None:
            A = np.hstack((A, x_extra))
        A = np.hstack((A, x))   
        return A


class LinearFitter(Fitter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def scale_and_fit(self, rms_x=False, log_x=False, log_y=False, check_cond_of_A=False, 
                            fit_mode='leastsq', regularization_lambda=0.0):
        self.log_x, self.log_y = log_x, log_y
        self.scale_y_values()
        self.x_scalar_train_scaled = self.scale_x(self.x_scalar_train)
        if self.x_extra is None:
            self.x_extra_train_scaled = None
        else:
            self.x_extra_train_scaled = self.scale_x(self.x_extra_train)
        self.A_train = self.construct_feature_matrix(self.x_scalar_train_scaled, self.y_current_train_scaled,
                                                     x_extra=self.x_extra_train_scaled,
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
        inverse_variances = 1/self.y_uncertainties_train_scaled**2
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
        self.x_scalar_test_scaled = self.scale_x(self.x_scalar_test)
        if self.x_extra_test is None:
            self.x_extra_test_scaled = None
        else:
            self.x_extra_test_scaled = self.scale_x(self.x_extra_test)
        self.A_test = self.construct_feature_matrix(self.x_scalar_test_scaled, self.y_current_test_scaled,
                                                    x_extra=self.x_extra_test_scaled)
        self.y_pred = self.predict_from_A(self.A_test)


    def predict(self, x, y_current, x_extra=None):
        x_scaled = self.scale_x(x)
        y_current_scaled = self.scale_y(y_current)
        A = self.construct_feature_matrix(x_scaled, y_current_scaled, x_extra=x_extra)
        y_pred = self.predict_from_A(A)
        return y_pred


    def predict_from_A(self, A):
        # A is assumed to be NOT "fitscaled" by x_fitscales
        # but A is "scaled": pre-scaled by other means, e.g. log
        y_pred_scaled = A @ self.theta
        y_pred = self.unscale_y(y_pred_scaled)
        return y_pred


    