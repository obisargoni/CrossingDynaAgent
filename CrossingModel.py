# Testing out accumulator model of crossing option choice

import numpy as np
import sys
import networkx as nx

from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from importlib import reload
import tilings
tilings = reload(tilings)
from tilings import TilingGroup


class MobileAgent(Agent):

    _loc = None
    _speed = None
    _bearing = None

    _loc_history = None

    def __init__(self, unique_id, model, l, s, b):
        super().__init__(unique_id, model)
        self._loc = l
        self._speed = s
        self._bearing = b

        self._loc_history = np.array([])

    def move(self):
        self._loc += (self._speed * np.sin(self._bearing), self._speed * np.cos(self._bearing))
        return

    def getSpeed(self):
        return self._speed

    def getBearing(self):
        return self._bearing

    def setBearing(self, b):
        self._bearing = b

    def getLoc(self):
        return self._loc



class Road(Agent):

    _length = None
    _width = None
    _nlanes = None

    _crossings = None
    _vehicles = None
    _buildings = None

    def __init__(self, unique_id, model, l, w, nl, xs = None, vs = None, blds = None):
        super().__init__(unique_id, model)
        self._length = l
        self._width = w
        self._nlanes = nl
        self._crossings = xs
        self._vehicles = vs
        self._buildings = blds

    def getVehicles(self):
        return self._vehicles

    def getLength(self):
        return self._length

    def getWidth(self):
        return self._width

    def getBuildings(self):
        return self._buildings

    def getCrossings(self):
        return self._crossings

    def getNLanes(self):
        return self._nlanes

class CrossingAlternative(Agent):

    _loc = None
    _ctype = None
    _name = None


    def __init__(self, unique_id, model, location = None, ctype = None, name = None):
        super().__init__(unique_id, model)
        self._loc = location
        self._ctype = ctype
        self._name = name

    def getLoc(self):
        return self._loc

    def getName(self):
        return self._name

    def getCrossingType(self):
        return self._ctype

class PedInternalModel():

    _mdp = None
    _s = None
    _terminal = False

    _crossing_features = None
    _dest_feature = None
    _vehicles = None

    def __init__(self, tg, cfs = None, df = None, vs = None):
        '''
        tg {TilingGroup} The tiling used to discretise the environment
        s {array} The feature vector corresponding to the agent's current state
        cfs {list} A list of arrays corresponding to all locations that correspond to crossing infrastructure
        df {array} The feature vector of the agent's destination
        vs {int} The number of vehicles on the road.
        '''
        
        # The tiling group used by the agent to discretise space
        self._tg = tg

        self._crossing_features = cfs
        self._dest_feature = df
        self._vehicles = vs

        self.build_mdp()

    def build_mdp(self):
        '''Given the tilings used to represent the space and the destination create an mdp representing the result of 'walk forward' and 'cross'
        actions in each tile. Tiles on the same side of the road as the destination only have walk forward actions available.
        '''
        # Get a lookup from node id to state
        tg_edgesx = np.array([t.edges[0] for t in self._tg.tilings]).flatten()
        tg_edgesx = np.concatenate((tg_edgesx, self._tg.limits[0]))
        tg_edgesx.sort()

        tg_edgesy = self._tg.limits[1]


        self.dict_id_feature = {}
        self._mdp = nx.DiGraph()

        # Connect features on one side of the road to each other
        edge_direction = '+'
        for iy, ey in enumerate(tg_edgesy):
            for ix, ex in enumerate(tg_edgesx):

                node_i = str(iy)+str(ix)
                node_j = str(iy) + str(ix+1)

                # Add lookup to dict
                f = self._tg.feature((ex, ey))

                self.dict_id_feature[node_i] = f

                # Add directed edge to graph if not at last edge
                if ix != len(tg_edgesx)-1:
                    # Depending on whist side of the road, connect in all same direction or all towards the destination feature
                    if edge_direction == '+':
                        self._mdp.add_edge(node_i, node_j, action = 'fwd')
                    else:
                        self._mdp.add_edge(node_j, node_i, action = 'fwd')                    
                else:
                    # Connect last edge to itself since can't go forward from here
                    self._mdp.add_edge(node_i,node_i, action = 'fwd')

                # Switch the edge direction if the starting node if the destination
                if np.equal(f,self._dest_feature).all():
                    edge_direction = '-'

        # Connect features across the road with cross actions
        for ix, ex in enumerate(tg_edgesx):
            node_i = str(0)+str(ix)
            node_j = str(1)+str(ix)
            self._mdp.add_edge(node_i, node_j, action = 'cross')


    @property
    def state(self):
        return self._s

    def step(self, a):
        '''Progress to new state following action a
        '''
        # Initialise the reward
        r = 0
        
        state_node = None
        # Loop through key value pais to find node corresponding to state
        for k,v in self.dict_id_feature.items():
            if np.equal(v,self._s).all():
                state_node = k
                break

        # Find node that results from taking this action
        for e in this._mdp.edges(nbunch=state_node, data='action'):
            if e[2] == a:
                r = self.reward(self._s, a)
                new_s = self.dict_id_feature[e[1]] 
                self._s = new_s
                break


        # If reached destination set terminal to true
        if np.equal(self._s, self._dest_feature).all():
            self._terminal = True

        return (self._s, r)

    def reward(self, s, a):
        '''Get the reward of arriving in state s
        '''

        if a == 'fwd':
            # -1 reward for each step taken forward regardless of state
            return -1
        elif a == 'cross':
            # Find out if ped is crossing by some crossing infrastructure
            cross_on_inf = s in self._crossing_features

            if cross_on_inf:
                return 0
            else:
                # Return value to reflect exposure of vehicles
                return -1*self.vehicles

    
    def setState(self, s):
        self._s = s
        self._terminal = False

    def isTerminal(self):
        return self._terminal


class Ped(MobileAgent):

    def __init__(self, unique_id, model, l, s, b, d):
        super().__init__(unique_id, model, l, s, b)
        self._dest = destination
        self._road_length = model.getRoad().getLength()

        # Initilise tiling group agent uses to discetise space
        ngroups = 2
        tiling_limits = [0,self._road_length]
        ntiles = [25]

        self._tg = TilingGroup(ngroups, tiling_limits, ntiles)

    def caLoc(self, ca):
        ca_loc = ca.getLoc()

        # Mid block crossings not assigned a locations because they take place at ped's current location
        if ca_loc is None:
            ca_loc = self._loc

        return ca_loc


    def ca_walk_time(self, ca):
        ca_loc = self.caLoc(ca)

        # separate costsing into waiting and walking on road time (no traffic exposure) time
        ww_time = abs(self._loc - ca_loc)/self._speed + abs(ca_loc - self._dest)/self._speed

        return ww_time

    def ca_detour_time(self, ca):
        '''The time difference between using the crossing alternative and walking directly to the destination
        via an unmarked crossing
        '''
        ca_loc = self.caLoc(ca)

        d_ca = abs(self._loc - ca_loc) + abs(ca_loc - self._dest)

        detour_dist = d_ca - abs(self._dest - self._loc)

        return detour_dist/self._speed


    def ca_salience_distances_to_dest(self):
        '''Salience of crossing option determined by difference between twice the road length and the distance to the destination
        '''
        ca_salience_distances = []
        for i,ca in enumerate(self._crossing_alternatives):

            # Get distance from agent to destination
            d = abs(self._dest - self._loc)

            # Get distnaces to and from the ca
            d_to = self.caLoc(ca) - self._loc
            d_from = self._dest - self.caLoc(ca)

            # Salience distance is difference between direct distance and distance via crossing, scaled by road length
            d_s = (2*self._road_length - (abs(d_to) + abs(d_from) - d)) / self._road_length
            ca_salience_distances.append(d_s)
        return np.array(ca_salience_distances)


    def ca_salience_factors_softmax(self, salience_type = 'ca'):
        if salience_type == 'ca':
            return scipy.special.softmax(self._lambda * self.ca_salience_distances_to_ca())
        else:
            return scipy.special.softmax(self._lambda * self.ca_salience_distances_to_dest())

    def step(self):

        # Check if ped has reached end of the road or if it has chosen a crossing
        if (self.getLoc() < self._road_length):

            self._loc_history = np.append(self._loc_history, self._loc)
            # move the ped along
            self.move()
        else:
            # When agent is done remove from schedule
            self.model.schedule.remove(self)

    def getDestination(self):
        return self._dest


class Vehicle(MobileAgent):

    def __init__(self, unique_id, model, l, s, b):
        super().__init__(unique_id, model, l, s, b)

    def step(self):

        # Check if ped has reached end of the road or if it has chosen a crossing
        if (self.getLoc() < self._road_length):

            self._loc_history = np.append(self._loc_history, self._loc)
            # move the agent along
            self.move()
        else:
            # When agent is done remove from schedule
            self.model.schedule.remove(self)



class CrossingModel(Model):
    def __init__(self, ped_origin, ped_destination, road_length, road_width, vehicle_flow, epsilon, gamma, ped_speed, lam, alpha, a_rate):
        self.schedule = RandomActivation(self)
        self.running = True
        self.nsteps = 0

        # Create two crossing alternatives, one a zebra crossing and one mid block crossing
        zebra_location = road_length * 0.75
        zebra_type = 'zebra'
        mid_block_type = 'unmarked'
        
        zebra = CrossingAlternative(0, self, location = zebra_location, ctype = zebra_type, name = 'z1', vehicle_flow = vehicle_flow)
        unmarked = CrossingAlternative(1, self, ctype = mid_block_type, name = 'mid1', vehicle_flow = vehicle_flow)

        # Crossing alternatives with salience factors
        crossing_altertives = np.array([unmarked,zebra])

        i = 0
        model_type = 'sampling'
        self.ped = Ped(i, self, location = ped_origin, speed = ped_speed, destination = ped_destination, crossing_altertives = crossing_altertives, road_length = road_length, road_width = road_width, epsilon = epsilon, gamma = gamma, lam = lam, alpha = alpha, a_rate = a_rate, model_type = model_type)
        self.schedule.add(self.ped)

        self.datacollector = DataCollector(agent_reporters={"CrossingType": "chosenCAType"})

        self.crossing_choice = None
        self.choice_step = None

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False
        self.nsteps += 1

