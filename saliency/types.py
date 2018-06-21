from enum import Enum

class SaliencyMethod(Enum):
    VANILLA = 1
    GUIDED = 2
    DECONV = 3
    TEST = 4
    PERTURBATION = 5
    #EXCITATION_BP = 6


class MapType(Enum):
    POSITIVE = 1
    NEGATIVE = 2
    ABSOLUTE = 3
    ORIGINAL = 4
    INPUT = 5
