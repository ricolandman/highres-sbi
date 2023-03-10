import numpy as np
from scipy.special import erfcinv

class Parameter():
    def __init__(self, name, prior):
        self.name = name
        self._prior = prior

    def prior(self, x):
        self.value = self._prior(x)
        return self.value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    def value_func(self, x):
        return lambda x: self.value


class ParameterSet():
    def __init__(self, parameters):
        self.parameters = parameters

    def prior(self, cube):
        prior = np.zeros_like(cube)
        for i, param in enumerate(self.parameters):
            prior[i] = param.prior(cube[i])
        return prior

    def param_dict(self, cube):
        params = {}
        for param, val in zip(self.parameters, cube):
            params[param.name] = val
        return params

    @property
    def N_params(self):
        return len(self.parameters)

    @property
    def names(self):
        names = [param.name for param in self.parameters]
        return names

    def add_params(self, parameter):
        if isinstance(parameter, list):
            for param in parameter:
                self.parameters.append(param)
        else:
            self.parameters.append(parameter)
    
    def sample(self, batch_size=None):
        '''
        print(batch_size, self.N_params)
        xs = np.random.uniform(0, 1, (batch_size, self.N_params))
        print(xs.shape)
        return np.array([self.prior(x) for x in xs])
        '''
        if batch_size is None: 
            x = np.random.uniform(0, 1, self.N_params)
            return self.prior(x)
        else:
            xs = np.random.uniform(0, 1, (self.N_params, batch_size))
            sample = np.array([param.prior(x) for param, x in zip(self.parameters, xs)]).T
            return sample

    @property
    def lower(self):
        lower_bounds = []
        for param in self.parameters:
            if hasattr(param._prior, 'lower'):
                lower_bounds.append(param._prior.lower)
            else:
                lower_bounds.append(np.nan)
        return np.array(lower_bounds)
    
    @property
    def upper(self):
        upper_bounds= []
        for param in self.parameters:
            if hasattr(param._prior, 'upper'):
                upper_bounds.append(param._prior.upper)
            else:
                upper_bounds.append(np.nan)
        return np.array(upper_bounds)


class uniform_prior():
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def __call__(self, x):
        return (self.upper - self.lower) * x + self.lower

'''
def gaussian_prior(mean, std):
    return lambda x: mean + std * np.sqrt(2) * erfcinv(2.0 * (1.0 - x))

def uniform_prior(start, end):
    return lambda x: (end-start) * x + start
'''