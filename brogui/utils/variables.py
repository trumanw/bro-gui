from abc import ABC
from enum import Enum

class ParameterType(Enum):
    continuous="continuous"
    categorical="categorical"
    ordinal="ordinal"

# Init viable acqucision functions
AF_OPTIONS = ["EI", "PI", "UCB", "qEI", "qPI", "qUCB", "qEHVI"]

# Init viable sampling strategies
DOE_OPTIONS = ["Full Factorial", "Latin Hypercube", "Norm Transform", "Randomized Design"]

# Init viable parameters
OPTION_PARAMS = ["RT", "T", "P", "GR", "LR"]
def VariableFactory(name="T"):

    """Factory Method"""
    parameters = {
        "RT": RT, "T": T, "P": P, "GR": GR, "LR": LR
    }

    return parameters[name]()

class Variable(ABC):
    def __init__(self):
        self.symbol = None
        self.parameter = None
        self.parameter_type = None
        self.unit = None
        self.parameter_range = None

class RT(Variable):
    def __init__(self):
        self.symbol = "RT"
        self.parameter = "Res. Time"
        self.parameter_type = ParameterType.continuous
        self.unit = 'min'
        self.parameter_range = [0, 100]
    
class T(Variable):
    def __init__(self):
        self.symbol = "T"
        self.parameter = "Temperature"
        self.parameter_type = ParameterType.continuous
        self.unit = "C"
        self.parameter_range = [10, 150]

class P(Variable):
    def __init__(self):
        self.symbol = "P"
        self.parameter = "Pressure"
        self.parameter_type = ParameterType.continuous
        self.unit = "bar"
        self.parameter_range = [0, 100]

class GR(Variable):
    def __init__(self):
        self.symbol = "GR"
        self.parameter = "Gas Flow Rate"
        self.parameter_type = ParameterType.continuous
        self.unit = "NmL/min"
        self.parameter_range = [0, 60]

class LR(Variable):
    def __init__(self):
        self.symbol = "LR"
        self.parameter = "Liquid FLow Rate"
        self.parameter_type = ParameterType.continuous
        self.unit = "mL/min"
        self.parameter_range = [0.3, 3]