# Figuring out how to represent pedestrians choice of crossing location by planning on an imagined network/state-action space

import os
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from importlib import reload
import tilings
tilings = reload(tilings)
from tilings import Tiling, TilingGroup

import CrossingModel
CrossingModel = reload(CrossingModel)
from CrossingModel import PedInternalModel, CrossingModel


###########################
#
#
# Testing tiling group
#
#
###########################

ngroups = 2
tiling_limits = [(0,50), (0, 10)]
ntiles = [25, 2]

tg = TilingGroup(ngroups, tiling_limits, ntiles)
dest = (30,10)
df = tg.feature(dest)


##############################
#
#
# Test setting up ped and road via model
#
#
##############################

# Road parameters
road_length = 100
road_width = 10
n_lanes = 1
vehicle_flow = 3

# Ped parameters
o = (0,0)
d = (road_length*0.5, road_width)
s = 1.5
b = 0
g = 0.7

cm = CrossingModel(road_length, road_width, vehicle_flow, n_lanes, o, d, g, s)