import pandas
import os

from enum import Enum

DATA_DIR = os.path.join('..','data')
EXAMPLE_DIR = os.path.join('..','zhusuan')

#######################################################################
class DataSets(Enum):
    EX0 = 'ex0'
    EX1 = 'ex1'
    EX1B = 'ex1b'
    EX2 = 'ex2'
    EX3 = 'ex3'
    EX4 = 'ex4'

#######################################################################
def data_dir():
    return DATA_DIR
def example_dir():
	return EXAMPLE_DIR
