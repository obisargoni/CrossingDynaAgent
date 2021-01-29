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

    _env = None
    _loc = None
    _speed = None
    _bearing = None

    _loc_history = None

    def __init__(self, unique_id, model, l, s, b):
        super().__init__(unique_id, model)
        self._env = model
        self._loc = l
        self._speed = s
        self._bearing = b

        self._loc_history = []

    def move(self):
        self._loc = (self._loc[0] + self._speed * np.sin(self._bearing), self._loc[1] + self._speed * np.cos(self._bearing))
        self._loc_history.append(self._loc)
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
        self._loc_history = []

class Road:

    _length = None
    _width = None
    _nlanes = None

    _crossings = None
    _vehicles = None
    _buildings = None

    def __init__(self, l, w, nl, xcoords = None, vf = None, blds = None):
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

class CrossingAlternative:

    _loc = None
    _ctype = None
    _name = None


    def __init__(self, location = None, ctype = None, name = None):
        self._loc = location
        self._ctype = ctype
        self._name = name

    def get_loc(self):
        return self._loc

    def getName(self):
        return self._name

    def getCrossingType(self):
        return self._ctype

class RoadEnv(Agent):

    _mdp = None

    _crossing_features = None
    _dest_feature = None
    _opp_dest_feature = None
    _road = None
    _s = None

    def __init__(self, unique_id, model, tg, road = None, destcoord = None, oppdestcoord = None):
        '''
        tg {TilingGroup} The tiling used to discretise the environment
        cfs {list} A list of arrays corresponding to all locations that correspond to crossing infrastructure
        destf {array} The feature vector of the agent's destination
        r {Road} The Road object in the road environment
        '''
        super().__init__(unique_id, model)

        # The tiling group used by the agent to discretise space
        self._tg = tg
        self._road = road

        self._crossing_features = [self._tg.feature(xcoord) for xcoord in self._road.getCrossingCoords()]
        self._dest_feature = self._tg.feature(destcoord)
        self._opp_dest_feature = self._tg.feature(oppdestcoord)

        self._sss = (self._tg.N, 2)

    def possible_actions(self):
        for a in (0,1):
            yield a

    def step(self, a):
        '''Given action taken by ped agent, update pedestrian's location, update state of the environment correspondingly, calculate the reward and return new 
        state and reward.
        '''
        # Initialise the reward
        r = 0

        # Update positions of agents based on this action
        self.model.ped.walk(a)

        # Constrain peds location to within the bounds of the road
        x = self.model.ped._loc[0] 
        y = self.model.ped._loc[1] 
        if x > self._road._length:
            x = self._road._length
        elif y > self._road._width:
            y = self._road._width
        elif y < 0:
            y = 0

        self.model.ped._loc = (x,y)

        # Use tilings to find new state
        new_state = self._tg.feature(self.model.ped._loc)

        # Calculate reward
        r = self.reward(new_state, a)

        # update road environment state
        self.set_state(new_state)

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
                return -1
            else:
                # Return value to reflect exposure of vehicles
                return -1*self._road.getVehicleFlow()

    def set_state_from_ped_location(self, loc):
        self._s = self._tg.feature(loc)

    def set_state(self, s):
        self._s = s

    def isTerminal(self):
        return np.equal(self._s, self._dest_feature).all()

    @property
    def state(self):
        return self._s

    @property
    def sss(self):
        return self._sss

class ModelRoadEnv:
    '''Class used to represent an agent's model of the environment, which they do planning with and update with real experience
    '''

    _model = None
    _rand = None
    _time = 1

    def __init__(self, rand = np.random):
        self._rand = rand
        self._model = dict()

    def step(self, s, a):
        '''Progress to new state following action a
        '''
        # Sample a next-state and reward from experience
        options = self._model[tuple(s)][a]
        choice_index = np.random.choice(len(options))
        next_state, reward, time = options[choice_index]

        next_state = np.array(next_state)

        return (next_state, reward)

    def update(self, state, action, next_state, reward):
        '''Update the model of the environment with experience from real environment

        #######################################################################
        # Copyright (C)                                                       #
        # 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
        # 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
        # Permission given to modify the code as long as you keep this        #
        # declaration at the top                                              #
        #######################################################################
        '''
        if tuple(state) not in self._model.keys():
            self._model[tuple(state)] = dict()

            # Initialise with actions so that planning can consider any available action from this state
            for action_ in [0,1]:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self._model[tuple(state)][action_] = [[list(state), 0, 1]]
                else:
                    self._model[tuple(state)][action_] = []


        self._model[tuple(state)][action].append([list(next_state), reward, self._time])
        self._time += 1

    def state_actions(self, state):
        '''Return the actions available to an agent from the input state
        '''

        if tuple(state) not in self._model.keys():
            return None
        else:
            return tuple(self._model[tuple(state)].keys())

class MDPModelRoadEnv:

    _s = None
    _sn = None
    _mdp = None
    _terminal_state = None
    dict_node_state = None

    def __init__(self, road_env):
        '''
        road_env {} The road environment this MDP model environment is a model of.
        '''
        self._terminal_state = road_env._dest_feature
        self.build_mdp(road_env)

    def build_mdp(self, road_env):
        '''Given the tilings used to represent the space and the destination create an mdp representing the result of 'walk forward' and 'cross'
        actions in each tile. Tiles on the same side of the road as the destination only have walk forward actions available.
        '''
        # Get a lookup from node id to state
        tg_edgesx = np.array([t.edges[0] for t in road_env._tg.tilings]).flatten()
        tg_edgesx = np.concatenate((tg_edgesx, road_env._tg.limits[0][:1])) # Only add in the lower limit
        tg_edgesx.sort()

        tg_edgesy = road_env._tg.limits[1]


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
                fi = road_env._tg.feature((exi, ey))
                fj = road_env._tg.feature((exj, ey))

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
                if np.equal(fj,road_env._dest_feature).all():
                    edge_direction = '-'

                    # Connect destination to itself
                    self._mdp.add_edge(node_j, node_j, action = 0)


        # Connect features across the road with cross actions
        for ix, ex in enumerate(tg_edgesx):
            node_i = str(0)+str(ix)
            node_j = str(1)+str(ix)

            # Action value of 1 corresponds to crossing. Only allow crossing in one direction
            self._mdp.add_edge(node_i, node_j, action = 1)


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

        return (self._s, r)

    def reward(self, s, a):
        '''Get the reward of arriving in state s
        '''
        # Need to use modelled reward, eg from experience

    def state_node(self, s):
        state_node = None
        # Loop through key value pais to find node corresponding to state
        for k,v in self.dict_node_state.items():
            if np.equal(v,s).all():
                state_node = k
                break
        return state_node
    
    def state_actions(self, s):
        sn = self.state_node(s)
        for (i,j,a) in self._mdp.edges(nbunch=sn, data='action'):
            yield a

    def state_actions(self):
        return state_actions(self._s)

    def set_state(self, s):
        self._s = s
        self._sn = self.state_node(s)

    def isTerminal(self):
        return np.equal(self._s, self._terminal_state).all()

    @property
    def state(self):
        return self._s

class Vehicle(MobileAgent):

    def __init__(self, unique_id, model, l, s, b):
        super().__init__(unique_id, model, l, s, b)

    def step(self):

        # Check if ped has reached end of the road or if it has chosen a crossing
        if (self.get_loc() < self._road_length):

            self._loc_history.append(self._loc)
            # move the agent along
            self.move()
        else:
            # When agent is done remove from schedule
            self.model.schedule.remove(self)

class Ped(MobileAgent):

    def __init__(self, unique_id, env, l, d, dd, b, s, g, alpha):
        '''
        unique_id {int} Unique ID used to index agent
        env {mesa.Model} The model environment agent is placed in
        l {tuple} The starting location of the agent
        d {tuple} The destination coordinate of the agent
        dd {tuple} The default destination coordiante of the agent
        b {tuple} The starting bearing of the agent
        s {double} The speed of the agent
        actions {tuple} The actions available to the agent
        g {double} The discount factor applied to future rewards by the agent
        a {double} Step size of update to feature vector weights
        '''
        super().__init__(unique_id, env, l, s, b)
        self._origin = l
        self.d = d
        self.dd = dd
        self._g = g
        self._alpha = alpha
        self._epsilon = 0.1
        self._nepochs = 0
        self._ntrainingsteps = 0
        self._max_steps_in_epoch = 500

        # initialise weights and n visits log
        self._w_log = np.array([])
        self._N_log = np.array([])

        # initialise value function
        self.reset_values(env.road_env.sss)

        # initialise model of env used for planning
        self.model_env = ModelRoadEnv(rand = np.random)
        #self.planning_policy = # Can refactor using decorators to return a planning policy function, but don't want to figure it out now


        # An alternative model of the environment using an mdp
        #self.model_env = MDPModelRoadEnv(env.road_env)
        #self.planning_policy = self.mdp_model_planning_policy(self.model_env, env.road_env.state, env.road_env._opp_dest_feature)

    def mdp_model_planning_policy(self, mdp_model, start_state, opp_destf):
        '''The agent plans using a planning policy and its model of the environment. The policy consists of the probability of taking actions
        move forward or cross road at each state. The policy probability is set such that the agent explores crossing close to its
        destination more frequently.

        Model search policy as negative binomial since two actions available at each state. For mean crossing location to be opposit destination
        need r=1 success after k failures, with k the number of times the agent has to continue straight between states.

        Args:
            opp_destf {array} Feature corresponding to the location opposite the agent's destination, where crossing takes the agent directly to its destination
        '''
        k=0
        while np.equal(mdp_model.state, opp_destf).all() == False:
            mdp_model.step(0)
            k+=1

            # Hack to avoid infinite loop when ped has walked past opp_dest_feature
            if k > len(mdp_model._mdp.nodes()):
                k = 0
                break

        p_cross = 1 / (k + 1)
        p_fwd = 1-p_cross

        # Reset the internal model state to the state the ped is currently in
        mdp_model.set_state(start_state)

        return (p_fwd, p_cross)

    def planning_policy_action(self, possible_actions, action_probabilities):
        a = None
        if len(possible_actions) == 1:
            a = possible_actions[0]
        else:
            p = [action_probabilities[a] for a in possible_actions]
            a = np.random.choice(possible_actions, p = p)
        return a

    def epsilon_greedy_action(self, state, possible_actions, epsilon):

        if np.random.rand() < epsilon:
            return self.greedy_action(state, possible_actions)
        else:
            return np.random.choice(possible_actions)


    def greedy_action(self, state, possible_actions):
        '''Find the action that have the highest associated value. Take that action
        '''
        q_max = -sys.maxsize
        a_chosen = None
        for a in possible_actions:
            q = self.q(state, a  = a)
            if q > q_max:
                a_chosen = a
                q_max = q
        return a_chosen

    def planning(self, start_state, nplanningsteps):
        s = start_state
        t = 0
        rtn = 0
        sa_visited = []
        for i in range(nplanningsteps):

            # Choose action
            possible_actions = self.model_env.state_actions(s)

            if possible_actions is None:
                continue

            a = self.epsilon_greedy_action(s, possible_actions, self._epsilon)


            # Sample next state and reward
            new_s, reward = self.model_env.step(s, a)
            sa_visited.append((s, a, reward))
            rtn += reward*(self._g**t) # Discount reward and add to total return
            t+=1
            s = new_s


        # Update value function with planning steps
        for i, (s, a, r) in enumerate(sa_visited):
            #print("{} : {}, {}".format(i,rtn,r))
            td_error = rtn - self.q(s, a = a)
            self.w[:, a] += self._alpha * td_error * s
            self.N[:,a] += s

            # Get return for next state
            rtn = (rtn - r) / self._g


    def dyna_step(self, nplanningsteps = 0):

        s = self._env.road_env.state

        # Do planning using internal model of env
        self.planning(s, nplanningsteps)

        # Now take greedy action
        possible_actions = list(self._env.road_env.possible_actions())
        a = self.epsilon_greedy_action(self._env.road_env.state, possible_actions, self._epsilon)
        next_state, r = self._env.road_env.step(a)

        # Update value function with real reward from env
        td_error = r - self.q(s, a = a)
        self.w[:, a] += self._alpha * td_error * s
        self.N[:,a] += s

        # Update internal model
        self.model_env.update(s, a, next_state, r)

    def train(self, nplanningsteps):
        '''Train the agent over multiple epochs
        '''
        self.dyna_step(nplanningsteps)
        self._ntrainingsteps += 1

        # if destination is reached reset environment and increase epoch count
        if (self._env.road_env.isTerminal() == True) | (self._ntrainingsteps > self._max_steps_in_epoch):
            self._loc = self._origin
            self._env.road_env.set_state_from_ped_location(self._loc)
            self._nepochs += 1
            self._ntrainingsteps = 0

    def step(self, nplanningsteps):
        '''Method to use when agent has been trained
        '''

        self.dyna_step(nplanningsteps)

        # When agent is done remove from schedule
        if (self._env.road_env.isTerminal() == False):
            self._env.schedule.remove(self)

    def walkdest(self, a, dest):
        
        vector_to_dest = [dest[i] - self._loc[i] for i in range(len(self._loc))]

        # a == 0 means walk toward destination, bearing either pi/2 or -pi/2. a == 1 means cross road, bearing = 0
        if a == 0:
            if (np.sign(vector_to_dest[0]) == -1):
                self._bearing = -np.pi/2
            else:
                self._bearing = np.pi/2
        elif a ==1:
            self._bearing = 0

        # move the ped along
        self.move()


    def walk(self, a):

        # Set whether to walk towards default destination or final destination
        if self._loc[1] > self._env.road_env._road.getWidth() / 2:
            dest = self.d
        else:
            dest = self.dd

        self.walkdest(a, dest)

    def reset_values(self, sss):
        # Initialise weights with arbitary low value so that agent doesn't take action it has not considered with internal model
        self._w = np.full(sss, -10.0)

        # Record number of times states visited
        self._N = np.zeros(sss)

    def q(self, s, a = None):
        '''Get value of state-action
        '''
        q = np.matmul(s, self._w)
        if a is None:
            return q
        else:
            return q[a]

    @property
    def N(self):
        return self._N

    @property
    def w(self):
        return self._w

    def log_values(self):
        self._w_log = np.append(self._w_log, self._w)
        self._N_log = np.append(self._N_log, self._N)


class CrossingModel(Model):
    def __init__(self, road_length, road_width, vehicle_flow, n_lanes, ped_origin, ped_destination, gamma, alpha, ped_speed):
        self.schedule = RandomActivation(self)
        self.running = True
        self.nsteps = 0

        # Create the road
        uid = 0
        self.init_road_env(uid, road_length, road_width, vehicle_flow, n_lanes, ped_destination)

        # Create the ped
        uid += 1
        bearing = 0

        ped_default_destination = (road_length, ped_origin[1])

        self.ped = Ped(uid, self, l = ped_origin, d = ped_destination, dd = ped_default_destination, b = bearing, s = ped_speed, g = gamma, alpha = alpha)
        self.road_env.set_state_from_ped_location(ped_origin)
        self.schedule.add(self.ped)

        self.datacollector = DataCollector(agent_reporters={"CrossingType": "chosenCAType"})

    def init_road_env(self, uid, road_length, road_width, vehicle_flow, n_lanes, ped_destination):
        '''Create the road env the pedestrian agent must learn to navigate
        '''
        crossing_start = int(road_length*0.75 - 2)
        crossing_end = int(road_length*0.75 + 2)
        crossing_coords = [(x,y) for x,y in itertools.product(range(crossing_start, crossing_end), range(0, road_width))]

        self.road = Road(road_length, road_width, n_lanes, xcoords = crossing_coords, vf = vehicle_flow, blds = None)

        # Initilise tiling group used to discetise space
        ngroups = 2
        tiling_limits = [(0,road_length), (0, road_width)]
        ntiles = [road_length, road_width]

        tg = TilingGroup(ngroups, tiling_limits, ntiles)

        # Note location opposite destination on agent's side of the road
        opp_dest = (ped_destination[0], -1*ped_destination[1])

        # Initialise an internal model of the street environment for the ped to use for planning
        self.road_env = RoadEnv(uid, self, tg, road = self.road, destcoord = ped_destination, oppdestcoord = opp_dest)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.get_agent_count() == 0:
            self.running = False
        self.nsteps += 1

    def getPed(self):
        return self.ped

    def getRoad(self):
        return self.road