# Testing out accumulator model of crossing option choice

import numpy as np
import sys
import networkx as nx
import itertools

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
        self._loc = (self._loc[0] + self._speed * np.sin(self._bearing), self._loc[1] + self._speed * np.cos(self._bearing))
        return

    def get_speed(self):
        return self._speed

    def get_bearing(self):
        return self._bearing

    def set_bearing(self, b):
        self._bearing = b

    def get_loc(self):
        return self._loc

    def set_loc(self, loc):
        '''
        loc {tuple} x,y of pedestrian
        '''
        self._loc = loc



class Road(Agent):

    _length = None
    _width = None
    _nlanes = None

    _crossings = None
    _vehicles = None
    _buildings = None

    def __init__(self, unique_id, model, l, w, nl, xcoords = None, vf = None, blds = None):
        super().__init__(unique_id, model)
        self._length = l
        self._width = w
        self._nlanes = nl
        self._crossing_coordinates = xcoords
        self._vehicles = vf
        self._buildings = blds

    def getVehicleFlow(self):
        return self._vehicles

    def getLength(self):
        return self._length

    def getWidth(self):
        return self._width

    def getBuildings(self):
        return self._buildings

    def getCrossingCoords(self):
        return self._crossing_coordinates

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

    def get_loc(self):
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

    def __init__(self, tg, cfs = None, destf = None, vs = None):
        '''
        tg {TilingGroup} The tiling used to discretise the environment
        s {array} The feature vector corresponding to the agent's current state
        cfs {list} A list of arrays corresponding to all locations that correspond to crossing infrastructure
        destf {array} The feature vector of the agent's destination
        vs {int} The number of vehicles on the road.
        '''
        
        # The tiling group used by the agent to discretise space
        self._tg = tg

        self._crossing_features = cfs
        self._dest_feature = destf
        self._vehicles = vs

        self._sss = (self._tg.N, 2)

        self._w_log = np.array([])
        self._N_log = np.array([])

        self.reset_values()

        self.build_mdp()

    def build_mdp(self):
        '''Given the tilings used to represent the space and the destination create an mdp representing the result of 'walk forward' and 'cross'
        actions in each tile. Tiles on the same side of the road as the destination only have walk forward actions available.
        '''
        # Get a lookup from node id to state
        tg_edgesx = np.array([t.edges[0] for t in self._tg.tilings]).flatten()
        tg_edgesx = np.concatenate((tg_edgesx, self._tg.limits[0][:1])) # Only add in the lower limit
        tg_edgesx.sort()

        tg_edgesy = self._tg.limits[1]


        self.dict_node_state = {}
        self._mdp = nx.DiGraph()

        # Connect features on one side of the road to each other
        edge_direction = '+'
        for iy in range(len(tg_edgesy)):
            for ix in range(len(tg_edgesx)-1):
                ey = tg_edgesy[iy]

                exi = tg_edgesx[ix]
                exj = tg_edgesx[ix+1]

                node_i = str(iy)+str(ix)
                node_j = str(iy) + str(ix+1)

                # Add lookup to dict
                fi = self._tg.feature((exi, ey))
                fj = self._tg.feature((exj, ey))

                # Duplicates adding states to dict but think this might be faster than checking
                self.dict_node_state[node_i] = fi
                self.dict_node_state[node_j] = fj

                # Add directed edge to graph if not at last edge
                if (ix == len(tg_edgesx)-2) & (iy == 0):
                    self._mdp.add_edge(node_j, node_j, action = 0)

                # Depending on whist side of the road, connect in all same direction or all towards the destination feature
                if edge_direction == '+':
                    self._mdp.add_edge(node_i, node_j, action = 0)
                else:
                    self._mdp.add_edge(node_j, node_i, action = 0)


                # Switch the edge direction if the starting node is the destination
                if np.equal(fj,self._dest_feature).all():
                    edge_direction = '-'

        # Connect features across the road with cross actions
        for ix, ex in enumerate(tg_edgesx):
            node_i = str(0)+str(ix)
            node_j = str(1)+str(ix)

            # Action value of 1 corresponds to crossing. Only allow crossing in one direction
            self._mdp.add_edge(node_i, node_j, action = 1)

    def state_node(self, s):
        state_node = None
        # Loop through key value pais to find node corresponding to state
        for k,v in self.dict_node_state.items():
            if np.equal(v,s).all():
                state_node = k
                break
        return state_node

    def state_actions(self):
        for (i,j,a) in self._mdp.edges(nbunch=self._sn, data='action'):
            yield a

    def state_action_values(self):
        for a in self.state_actions():
            yield (a, self.q(self._s, a))

    def step(self, a):
        '''Progress to new state following action a
        '''
        # Initialise the reward
        r = 0

        # Find node that results from taking this action
        for e in self._mdp.edges(nbunch=self._sn, data='action'):
            if e[2] == a:
                r = self.reward(self._s, a)
                self._sn = e[1]
                self._s = self.dict_node_state[self._sn] 
                break

        # If reached destination set terminal to true
        if np.equal(self._s, self._dest_feature).all():
            self._terminal = True

        return (self._s, r)

    def reward(self, s, a):
        '''Get the reward of arriving in state s
        '''

        if a == 0:
            # -1 reward for each step taken forward regardless of state
            return -1
        elif a == 1:
            # Find out if ped is crossing by some crossing infrastructure
            cross_on_inf = np.array([np.equal(s, cf).all() for cf in self._crossing_features]).any()

            if cross_on_inf:
                return 0
            else:
                # Return value to reflect exposure of vehicles
                return -1*self._vehicles

    def q(self, s, a = None):
        '''Get value of state-action
        '''
        q = np.matmul(s, self._w)
        if a is None:
            return q
        else:
            return q[a]

    def set_state(self, s):
        self._s = s
        self._sn = self.state_node(s)
        self._terminal = False

    def isTerminal(self):
        return self._terminal

    def log_values(self):
        self._w_log = np.append(self._w_log, self._w)
        self._N_log = np.append(self._N_log, self._N)


    def reset_values(self):
        # Initialise weights with arbitary low value so that agent doesn't take action it has not considered with internal model
        self._w = np.full(self._sss, -10.0)

        # Record number of times states visited
        self._N = np.zeros(self._sss)

    @property
    def state(self):
        return self._s

    @property
    def w(self):
        return self._w

    @property
    def sss(self):
        return self._sss
    
    @property
    def N(self):
        return self._N



class Ped(MobileAgent):

    def __init__(self, unique_id, model, l, b, s, d, g, a):
        '''
        unique_id {int} Unique ID used to index agent
        model {mesa.Model} The model environment agent is placed in
        l {tuple} The starting location of the agent
        b {tuple} The starting bearing of the agent
        s {double} The speed of the agent
        d {tuple} The destination of the agent
        g {double} The discount factor applied to future rewards by the agent
        a {double} Step size of update to feature vector weights
        '''
        super().__init__(unique_id, model, l, s, b)
        self._dest = d
        self._road_length = model.getRoad().getLength()
        self._road_width = model.getRoad().getWidth()
        self._crossing_coordinates = model.getRoad().getCrossingCoords()
        self._g = g
        self._a = a

        # Note location opposite destination on agent's side of the road
        self._opp_dest = (self._dest[0], self._loc[1])

        # Initilise tiling group agent uses to discetise space
        ngroups = 2
        tiling_limits = [(0,self._road_length), (0, self._road_width)]
        ntiles = [25, 2]

        self._tg = TilingGroup(ngroups, tiling_limits, ntiles)

        self._dest_feature = self._tg.feature(self._dest)
        self._crossings_features = [self._tg.feature(c) for c in self._crossing_coordinates]
        self._opp_dest_feature = self._tg.feature(self._opp_dest)

        # Initialise an internal model of the street environment for the ped to use for planning
        self.internal_model = PedInternalModel(self._tg, cfs = self._crossings_features, destf = self._dest_feature, vs = model.getRoad().getVehicleFlow())


    def set_search_policy(self, ped_locf, opp_destf):
        '''The agent explores its internal model using a search policy. The policy consists of the probability of taking actions
        move forward or cross road at each state. The policy probability is set such that the agent explores crossing close to its
        destination more frequently.

        Model search policy as negative binomial since two actions available at each state. For mean crossing location to be opposit destination
        need r=1 success after k failures, with k the number of times the agent has to continue straight between states.

        Args:
            ped_locf {array} Feature corresponding to the pedestrian agent's current location
            opp_destf {array} Feature corresponding to the location opposite the agent's destination, where crossing takes the agent directly to its destination
        '''

        self.internal_model.set_state(ped_locf)

        k=0
        while np.equal(self.internal_model._s, opp_destf).all() == False:
            self.internal_model.step(0)
            k+=1

            # Hack to avoid infinite loop when ped has walked past opp_dest_feature
            if k > self._tg.N:
                k = 0
                break

        p_cross = 1 / (k + 1)
        p_fwd = 1-p_cross

        # Reset the internal model state to the state the ped is currently in
        self.internal_model.set_state(ped_locf)

        self.search_policy = (p_fwd, p_cross)

    def choose_search_action(self, possible_actions, action_probabilities):
        a = None
        if len(possible_actions) == 1:
            a = possible_actions[0]
        else:
            p = [action_probabilities[a] for a in possible_actions]
            a = np.random.choice(possible_actions, p = p)
        return a

    def greedy_action(self):
        '''Find the action that have the highest associated value. Take that action
        '''
        q_max = -sys.maxsize
        a_chosen = None
        for a, q in self.state_action_values():
            if q > q_max:
                a_chosen = a
        return a

    def run_episode_return_states_actions_total_return(self, policy):
        # Run episode, get states and actions visited and total discounted return
        # This currently works out as every visit monte carlo update
        sa_visited = []
        rtn = 0
        t = 0
        start_state = self.internal_model._s
        while self.internal_model._terminal == False:
            s = self.internal_model._s

            # get available actions
            possible_actions = list(self.internal_model.state_actions())

            # choose action using policy
            a = self.choose_search_action(possible_actions, policy)
            sa_visited.append((s, a))

            # take action
            new_s, reward = self.internal_model.step(a)
            rtn += reward*(self._g**t) # Discount reward and add to total return
            t+=1

        # Resect internal model to starting state after episode
        self.internal_model.set_state(start_state)

        return (sa_visited, rtn)

    def mc_update_of_internal_model(self):
        '''Help inform what action to take now by exploring the internal model of the road environment, using it to plan
        best action at current state.
        '''

        # Set the current state of the internal model as the ped's current location
        loc_feature = self._tg.feature(self._loc)

        # Set search policy. This method sets the current state of ped's internal model to be state corresponding to ped's current location
        self.set_search_policy(loc_feature,self._opp_dest_feature)

        sa_visited, rtn = self.run_episode_return_states_actions_total_return(self.search_policy)

        # Use this to update weights using the return
        for s,a in sa_visited:
            td_error = rtn - self.internal_model.q(s, a = a)
            self.internal_model.w[:, a] += self._a * td_error * s
            self.internal_model.N[:,a] += s


    def step(self):

        # Check if ped has reached its destination
        if (self.internal_model.isTerminal() == False):
            # Run MC update a certain number of times -  this is the deliberation before next step
            for i in range(10):
                self.mc_update_of_internal_model()

            # Now choose greedy action
            a = self.greedy_action()

            self.walk(a)

        else:
            # When agent is done remove from schedule
            self.model.schedule.remove(self)

    def walk(self, a):
        '''
        Given an action, determine the direction of movement and move in that direction
        '''
        vector_to_dest = [self._dest[i] - self._loc[i] for i in range(len(self._loc))]

        # a == 0 means walk toward destination, bearing either pi/2 or -pi/2. a == 1 means cross road, bearing = 0
        if a == 0:
            if np.sign(vector_to_dest[0]) == -1:
                self._bearing = -np.pi/2
            else:
                self._bearing = np.pi/2
        elif a ==1:
            self._bearing = 0

        self._loc_history = np.append(self._loc_history, self._loc)
        
        # move the ped along
        self.move()


    def getDestination(self):
        return self._dest

    def set_dest(self, d):
        self._dest = d


class Vehicle(MobileAgent):

    def __init__(self, unique_id, model, l, s, b):
        super().__init__(unique_id, model, l, s, b)

    def step(self):

        # Check if ped has reached end of the road or if it has chosen a crossing
        if (self.get_loc() < self._road_length):

            self._loc_history = np.append(self._loc_history, self._loc)
            # move the agent along
            self.move()
        else:
            # When agent is done remove from schedule
            self.model.schedule.remove(self)



class CrossingModel(Model):
    def __init__(self, road_length, road_width, vehicle_flow, n_lanes, ped_origin, ped_destination, gamma, alpha, ped_speed):
        self.schedule = RandomActivation(self)
        self.running = True
        self.nsteps = 0

        # Create the road
        uid = 0

        crossing_start = int(road_length*0.75 - 2)
        crossing_end = int(road_length*0.75 + 2)
        crossing_coords = [(x,y) for x,y in itertools.product(range(crossing_start, crossing_end), [0,road_width])]

        self.road = Road(uid, self, road_length, road_width, n_lanes, xcoords = crossing_coords, vf = vehicle_flow, blds = None)

        # Create the ped
        uid += 1
        bearing = 0

        self.ped = Ped(uid, self, l = ped_origin, b = bearing, s = ped_speed, d = ped_destination, g = gamma, a = alpha)
        self.schedule.add(self.ped)

        self.datacollector = DataCollector(agent_reporters={"CrossingType": "chosenCAType"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False
        self.nsteps += 1

    def getRoad(self):
        return self.road

    def getPed(self):
        return self.ped

