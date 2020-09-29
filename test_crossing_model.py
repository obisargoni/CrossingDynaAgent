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
a = 0.05

cm = CrossingModel(road_length, road_width, vehicle_flow, n_lanes, o, d, g, a, s)

# Get ped and road

ped = cm.getPed()
road = cm.getRoad()

loc_feature = ped._tg.feature(ped._loc)


# Writing up tests of the MDP


# Test crossig opposite destination results in terminal state

ped.set_search_policy(loc_feature, ped._opp_dest_feature)

# get number of foward steps before opposite dest
k = int((1/ped.search_policy[1][1]) - 1)

actions = [0] * k
for a in actions:
	ped.internal_model.step(a)
np.equal(ped.internal_model._s, ped._opp_dest_feature).all()

ped.internal_model.step(1)
assert ped.internal_model.isTerminal()


# Test going past opposite destination by 2 steps and crossing requires 2 steps to get to dest
ped.internal_model.setState(loc_feature)
actions = [0] * (k+2)
for a in actions:
	ped.internal_model.step(a)

ped.internal_model.step(1)
assert ped.internal_model.isTerminal() == False
ped.internal_model.step(0)
assert ped.internal_model.isTerminal() == False
ped.internal_model.step(0)
assert ped.internal_model.isTerminal() == True


# Test crossing straight away and walking ahead requires k steps after crossing
ped.internal_model.setState(loc_feature)

ped.internal_model.step(1)

n= 0
while ped.internal_model.isTerminal() == False:
	ped.internal_model.step(0)
	n+=1
assert n == k


# Test reaching end of starting side of road is stuck on same state if continues to take forward action
ped.internal_model.setState(loc_feature)
end_road_feature = ped._tg.feature((100,0))
end_state_node = ped.internal_model.state_node(end_road_feature)
nsteps = int(end_state_node)
actions = [0] * (nsteps + 2)
for a in actions:
	ped.internal_model.step(a)
ped.internal_model.step(1)

n = 0
while ped.internal_model.isTerminal() == False:
	ped.internal_model.step(0)
	n+=1
assert n == (nsteps - k) - 1

