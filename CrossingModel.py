# Testing out accumulator model of crossing option choice

import numpy as np
import scipy.special
from scipy.stats import bernoulli
import sys

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

class Ped(MobileAgent):

    _dest = None

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

