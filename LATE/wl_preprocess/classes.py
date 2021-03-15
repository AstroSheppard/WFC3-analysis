import numpy as np
from scipy.stats import norm

class MargPriors:
    """ Class to hold priors for each parameter."""
    def __init__(self, labels):
        """ Initialize prior array with prior bounds that
        are almost never changed. This includes uninformative 
        systematic parameters, which will never have a physically
        motivated priors, and a loose event depth prior. Center
        of event time, inclination, a/rs, and the linear limb
        darkening coefficient need to be set if being fit for. 
        """
        self.priors = np.empty((32, 3), dtype='object')
        self.parameter_names = labels
        self.add_default_systematic_priors()

    def set_systematic_model(self, sys_model):
        """Set the systematic model, which is a 1-d array. Each 
        element in the array is a 0 if the corresponding parameter
        is open in fitting, and a 1 if it is fixed.
        """
        self.systematic_model = sys_model
        self.open_parameters = self.parameter_names[sys_model==0]
        self.priors[sys_model==1, 0] = None

    def add_default_systematic_priors(self):
        self.add_uniform_prior('Depth', 0.0, 0.2)
        self.add_log_uniform_prior('HST1', -6, 7)
        self.add_log_uniform_prior('HST2', -6, 7)
        self.add_log_uniform_prior('HST3', -6, 7)
        self.add_log_uniform_prior('HST4', -6, 7)
        self.add_log_uniform_prior('sh1', -6, 7)
        self.add_log_uniform_prior('sh2', -6, 7)
        self.add_log_uniform_prior('sh3', -6, 7)
        self.add_log_uniform_prior('sh4', -6, 7)
        self.add_uniform_prior('Eclipse Depth', 0, 0.2)
        self.add_uniform_prior('fnorm', 0.5, 3.0)
        self.add_uniform_prior('rnorm', 0.5, 3.0)
        self.add_uniform_prior('flinear', -10.0, 10.0)
        self.add_uniform_prior('rlinear', -10.0, 10.0)
        self.add_uniform_prior('fquad', -10.0, 10.0)
        self.add_uniform_prior('rquad', -10.0, 10.0)
        self.add_log_uniform_prior('fexpb', -4, 3)
        self.add_log_uniform_prior('flogb', -4, 3)
        self.add_log_uniform_prior('rexpb', -4, 3)
        self.add_log_uniform_prior('rlogb', -4, 3)
        self.add_uniform_prior('fexpc', -10.0, 10.0)
        self.add_uniform_prior('flogc', -10.0, 10.0)
        self.add_uniform_prior('rexpc', -10.0, 10.0)
        self.add_uniform_prior('rlogc', -10.0, 10.0)

    def set_initial_guesses(self, initial_guess):
        self.initial_guess = initial_guess

    def set_initial_errors(self, initial_error):
        """ Uncertainty in each parametereither from literature
        or from KMPFIT. Used to generate positions of ball of 
        walkers.
        """
        self.initial_error = initial_error

    def add_uniform_prior(self, parameter, lower, upper):
        ix = np.where(self.parameter_names==parameter)[0][0]
        self.priors[ix, 0] = 'uniform'
        self.priors[ix, 1] = lower
        self.priors[ix, 2] = upper

    def add_log_uniform_prior(self, parameter, log_lower, log_upper):
        """ Method to add log-uniform prior range. For range between 0.01 
        and 1000, bounds would be -2 and 3.
        """
        ix = np.where(self.parameter_names==parameter)[0][0]
        self.priors[ix, 0] = 'log-uniform'
        self.priors[ix, 1] = log_lower
        self.priors[ix, 2] = log_upper
    
    def add_gaussian_prior(self, parameter, mean, std):
        ix = np.where(self.parameter_names==parameter)[0][0]
        self.priors[ix, 0] = 'gaussian'
        self.priors[ix, 1] = mean
        self.priors[ix, 2] = std

    def prior_check(self, parameter_name, parameter_value):
        ix = np.where(self.parameter_names==parameter_name)[0][0]
        prior = self.priors[ix,:]
        prior_type = prior[0]
        if prior_type=='uniform':
            return prior[1] < parameter_value < prior[2]
        elif prior_type=='log-uniform':
            log_parameter = np.log10(np.abs(parameter_value))
            return prior[1] < log_parameter < prior[2]
        elif prior_type=='gaussian':
            return norm.pdf(parameter_value, prior[1], prior[2])
        else:
            raise ValueError('Need to set prior for ' \
                             + parameter_name + '.')

    def check_initial_positions(self, positions):
        in_bounds = np.ones(positions.shape[1])
        for i, p_initials in enumerate(positions.T):
            for p_value in p_initials:
                in_bound = self.prior_check(self.open_parameters[i], p_value)
                if in_bound == False:
                    in_bounds[i] = 0
        out_of_bounds = self.open_parameters[in_bounds==0]
        if len(out_of_bounds)>0:
            raise ValueError('Initial position of at least ' \
                             'one walker for parameters %s is ' \
                             'outside prior limits.' % out_of_bounds)

    def convert_to_bounds(self):
        priors = self.priors
        p0 = self.initial_guess
        parameter_names = self.parameter_names
        bounds = []
        for i, prior in enumerate(priors):
            if prior[0]=='uniform':
                bound = (prior[1], prior[2])
                bounds.append(bound)
            if prior[0]=='gaussian':
                lower = prior[1] - 5*prior[2]
                upper = prior[1] + 5*prior[2]
                if parameter_names[i] == 'i':
                    upper = 90.
                bound = (lower, upper)
                bounds.append(bound)
            if prior[0]=='log-uniform':
                bound = (None, None)
                bounds.append(bound)
            if prior[0]==None:
                bound = (p0[i], p0[i])
                bounds.append(bound)
        self.bounds = tuple(bounds)
        return self.bounds
