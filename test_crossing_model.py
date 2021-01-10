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
from CrossingModel import MDPModelRoadEnv, CrossingModel


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

loc_feature = cm.road_env._tg.feature(ped._loc)


# Writing up tests of the MDP


# Test crossig opposite destination results in terminal state

ped.model_env = MDPModelRoadEnv(cm.road_env)
ped.planning_policy = ped.mdp_model_planning_policy(ped.model_env, cm.road_env.state, cm.road_env._opp_dest_feature)

# get number of foward steps before opposite dest
k = int((1/ped.planning_policy[1]) - 1)

actions = [0] * k
for a in actions:
	ped.model_env.step(a)
np.equal(ped.model_env._s, cm.road_env._opp_dest_feature).all()

ped.model_env.step(1)
assert ped.model_env.isTerminal()


# Test going past opposite destination by 2 steps and crossing requires 2 steps to get to dest
ped.model_env.set_state(loc_feature)
actions = [0] * (k+2)
for a in actions:
	ped.model_env.step(a)

ped.model_env.step(1)
assert ped.model_env.isTerminal() == False
ped.model_env.step(0)
assert ped.model_env.isTerminal() == False
ped.model_env.step(0)
assert ped.model_env.isTerminal() == True


# Test crossing straight away and walking ahead requires k steps after crossing
ped.model_env.set_state(loc_feature)

ped.model_env.step(1)

n= 0
while ped.model_env.isTerminal() == False:
	ped.model_env.step(0)
	n+=1
assert n == k


# Test reaching end of starting side of road is stuck on same state if continues to take forward action
ped.model_env.set_state(loc_feature)
end_road_feature = cm.road_env._tg.feature((100,0))
end_state_node = ped.model_env.state_node(end_road_feature)
nsteps = int(end_state_node)
actions = [0] * (nsteps)
for a in actions:
	ped.model_env.step(a)

# Should now be at end of road
s = ped.model_env.state

# Take some more steps forward
for i in range(3):
	ped.model_env.step(0)

assert np.equal(s, ped.model_env.state).all()

# Now cross and see how many steps required to get to dest
ped.model_env.step(1)

n = 0
while ped.model_env.isTerminal() == False:
	ped.model_env.step(0)
	n+=1
assert n == (nsteps - k)


################################
#
#
# test direction of ped movement given action
#
#
################################

l = (0,0)
ped.set_loc(l)
dest = (0,1)
a = 0
ped._walk(a, dest)
assert ped._loc[0] > l[0]

l = (3,0)
ped.set_loc(l)
dest = (5,1)
ped._walk(a, dest)
assert ped._loc[0] > l[0]

l = (3,0)
ped.set_loc(l)
dest = (1,1)
ped._walk(a, dest)
assert ped._loc[0] < l[0]

l = (3,0)
ped.set_loc(l)
dest = (1,1)
a = 1
ped._walk(a, dest)
assert ped._loc[0] == l[0]
assert ped._loc[1] > l[1]