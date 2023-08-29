import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, MultiDiscrete
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class RtGridEnv(MultiAgentEnv):
    def __init__(self, length: int, width: int, nets: list, macros: list, edge_capacity: np.ndarray, max_step: int = 1000):
        """
        Args:
            length (int): length of the canvas
            width (int): width of the canvas
            nets (list): a list of nets to be routed
            macros (list): a list of macros that has been placed on the canvas by placement
        """
        self.length = length
        self.width = width
        self.nets = nets
        self.n_nets = len(self.nets)
        self.macros = macros
        self.initial_capacity = edge_capacity.copy()
        self.initial_capacity.setflags(write=False)
        self.edge_capacity = edge_capacity.copy()
        self.max_capacity = np.max(self.edge_capacity) + 1  # plus one to account for the behavior of gym.MultiDiscrete
        self.max_step = max_step
        self.step_counter = 0  # counts the number of steps elapsed for the current episode

        self.agents_id = []
        for i in range(self.n_nets):
            self.agents_id.append("agent_{}".format(i))
        self._agent_ids = set(self.agents_id)
        self.state = {}
        self.agent_position = {}
        self.goal_position = {}
        self.change_pin_flag = self.reset_flags({})
        self.done_flag = self.reset_flags({})
        # the done flag needs an additional "__all__" key to indicate all agents are done
        self.done_flag["__all__"] = False
        self.pin_counter = {}
        self.reset_pin_counters()
        self.path_x = self.generate_path(self.nets)
        self.path_y = self.generate_path(self.nets)
        self.decomposed_nets = {}
        for i in range(self.n_nets):
            self.decomposed_nets[self.agents_id[i]] = self.prim_mst(self.nets[i])

        # sort the decomposed wires by their Manhattan distances such that longer 2-pin wires will be routed first
        for agent, routing_dict in self.decomposed_nets.items():
            self.sort_wires_by_distance(routing_dict)

        # initialize the agent to route the first 2-pin net decomposed from the first multi-pin net
        for agent_id in self.agents_id:
            self.update_positions(agent_id)
            self.update_path(agent_id)

        self.action_space = Discrete(4)
        self.observation_space = MultiDiscrete([self.length, self.width, self.length, self.width, 2, 2, 2, 2])

    def sort_wires_by_distance(self, agent_routing):
        u_coords = agent_routing['u']
        v_coords = agent_routing['v']

        sorted_indices = sorted(range(len(u_coords)), key=lambda i: self.manhattan_distance(u_coords[i], v_coords[i]), reverse=True)

        sorted_u_coords = [u_coords[i] for i in sorted_indices]
        sorted_v_coords = [v_coords[i] for i in sorted_indices]

        agent_routing['u'] = sorted_u_coords
        agent_routing['v'] = sorted_v_coords

    def update_positions(self, agent_id: str):
        """
        Update the agent position with the starting pin of the next 2-pin net.
        Update the goal position with the new goal.
        """
        self.agent_position[agent_id] = np.array(self.decomposed_nets[agent_id]['u'][self.pin_counter[agent_id]])
        self.goal_position[agent_id] = np.array(self.decomposed_nets[agent_id]['v'][self.pin_counter[agent_id]])

    def update_path(self, agent_id: str):
        """Update the path agent has traveled."""
        self.path_x[agent_id][self.pin_counter[agent_id]].append(self.agent_position[agent_id][0])
        self.path_y[agent_id][self.pin_counter[agent_id]].append(self.agent_position[agent_id][1])

    def reset_pin_counters(self):
        """Set the pin counter of each agent to 0."""
        for i in range(self.n_nets):
            self.pin_counter[self.agents_id[i]] = 0

    def reset_flags(self, flags: dict):
        for i in range(self.n_nets):
            flags[self.agents_id[i]] = False

        return flags

    def generate_path(self, nets: list):
        """Generate the list data structure to hold the path traveled by the agent."""
        path = {}
        for i in range(len(nets)):
            path[self.agents_id[i]] = []
            for j in range(len(nets[i])-1):
                path[self.agents_id[i]].append([])

        return path

    def manhattan_distance(self, p1: tuple, p2: tuple):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def prim_mst(self, pins):
        """
        Compute the Minimum Spanning Tree (MST) using Prim's algorithm.

        Args:
            pins (list): List of (x, y) coordinates representing the pin locations.

        Returns:
            dict: a dictionary containing the vertices of all the edges in the MST

        Note:
            - The pins list should contain at least two points.
        """

        distances = {}
        for i in range(len(pins)):
            for j in range(i+1, len(pins)):
                p1 = pins[i]
                p2 = pins[j]
                distances[(i, j)] = self.manhattan_distance(p1, p2)
                distances[(j, i)] = distances[(i, j)]  # Add symmetric distance

        # Initialize
        num_pins = len(pins)
        visited = [False] * num_pins
        mst_u = []
        mst_v = []
        start_vertex = 0
        visited[start_vertex] = True

        # Create a priority queue
        pq = []

        # Mark the initial vertex as visited
        for i in range(num_pins):
            if i != start_vertex:
                heapq.heappush(pq, (distances[(start_vertex, i)], start_vertex, i))

        # Update the priority queue and perform Prim's algorithm
        while pq:
            if (len(mst_u) == len(pins) - 1):  # for n pins, the MST should at most have n-1 edges
                break

            weight, u, v = heapq.heappop(pq)

            if visited[v]:
                # print(f"Skipping edge: {weight} - {u} - {v}")
                continue

            # Prim's algorithm iteration
            visited[v] = True
            mst_u.append(pins[u])
            mst_v.append(pins[v])

            for i in range(num_pins):
                if not visited[i]:
                    heapq.heappush(pq, (distances[(v, i)], v, i))

        mst = {'u': mst_u, 'v': mst_v}

        return mst

    def update_capacity(self, agent_position: np.ndarray, action: int):
        """
        Update the edge capacities after taking an action
        """

        # reduce the capacity of the current node
        self.edge_capacity[agent_position[0]][agent_position[1]][action] += -1

        # reduce the capacity of the next node's corresponding edge
        new_node = self.compute_new_position(agent_position, action)
        corresponding_edge = (action + 2) % 4
        self.edge_capacity[new_node[0]][new_node[1]][corresponding_edge] += -1

    def compute_new_position(self, agent_position: np.ndarray, action: int):
        """
        Compute new agent positions
        """
        if action == 0:  # up
            new_position = (agent_position[0], agent_position[1]+1)
        elif action == 1:  # right
            new_position = (agent_position[0]+1, agent_position[1])
        elif action == 2:  # down
            new_position = (agent_position[0], agent_position[1]-1)
        elif action == 3:  # left
            new_position = (agent_position[0]-1, agent_position[1])

        return new_position

    def check_move_validity(self, agent_position: np.ndarray, action: int):
        """
        Check whether a move is valid by checking:
        (1) the edge the move is about to use has capacity greater than 0
        (2) the position after the move is not within macro regions
        (3) the position after the move in within in the routing canvas
        Returns True if the move is valid, False if invalid
        """
        # capacity of the 4 neighboring edges of the current agent position
        node_capacity = self.edge_capacity[agent_position[0]][agent_position[1]]

        new_position = self.compute_new_position(agent_position, action)

        macro_flag = new_position not in self.macros
        bound_flag = new_position[0] in range(self.length) and new_position[1] in range(self.width)
        capacity_flag = node_capacity[action] > 0

        valid = macro_flag and bound_flag and capacity_flag

        return valid

    def radar(self, agent_position: np.ndarray):
        """Tell the agent, given its current position, which of the 4 directions it can choose"""
        result = []
        for action in range(0, 4):
            result.append(int(self.check_move_validity(agent_position, action)))

        return np.array(result)

    def step(self, action: dict):
        # extract all the active agents in this time step
        active_agent = list(action.keys())
        # used for computing observations
        initial_active_agent = list(action.keys())
        reward = {}

        # if we have reached our maximum time step, set the all done flag
        self.step_counter += 1
        if self.step_counter >= self.max_step:
            for agent_id in active_agent:
                reward[agent_id] = -1
            self.done_flag["__all__"] = True
            observation = {agent: self.state[agent] for agent in initial_active_agent if agent in self.state}
            return observation, reward, self.done_flag, {}

        # update pins for those agents in need
        # Only agents that have the change pin flag set to True AND are active, will be updated
        pin_flag_agents = [key for key, value in self.change_pin_flag.items() if value]
        pin_flag_agents = list(set(active_agent).intersection(set(pin_flag_agents)))
        for agent_id in pin_flag_agents:
            self.update_positions(agent_id)
            self.update_path(agent_id)
            reward[agent_id] = 0
            self.change_pin_flag[agent_id] = False
            self.state[agent_id] = np.concatenate([
                self.agent_position[agent_id],
                self.goal_position[agent_id],
                self.radar(self.agent_position[agent_id])
                ])
            active_agent.remove(agent_id)  # de-active agents that undergoes pin-upgrading, such that they won't be unintentionally accessed

        for agent_id in active_agent:
            if self.check_move_validity(self.agent_position[agent_id], action[agent_id]):
                self.update_capacity(self.agent_position[agent_id], action[agent_id])
                self.agent_position[agent_id] = np.array(list(self.compute_new_position(self.agent_position[agent_id], action[agent_id])))
                self.update_path(agent_id)

            if np.array_equal(self.agent_position[agent_id], self.goal_position[agent_id]):
                reward[agent_id] = 10
                self.update_counters(agent_id)
            else:
                reward[agent_id] = -1

            self.state[agent_id] = np.concatenate([
                self.agent_position[agent_id],
                self.goal_position[agent_id],
                self.radar(self.agent_position[agent_id])
                ])
        # if all agents are done, set the __all__ flag
        self.done_flag["__all__"] = all(self.done_flag[agent_key] for agent_key in self.done_flag if agent_key.startswith('agent_'))
        # only return observations for active agents
        observation = {agent: self.state[agent] for agent in initial_active_agent if agent in self.state}

        return observation, reward, self.done_flag, {}

    def update_counters(self, agent_id: str):
        # one 2-pin net within one multi-pin net is done
        self.pin_counter[agent_id] += 1
        self.change_pin_flag[agent_id] = True
        net_id = int(agent_id.split("_")[1])

        if self.pin_counter[agent_id] == len(self.nets[net_id]) - 1:
            # this agent is done, it has routed all the pins
            self.change_pin_flag[agent_id] = False
            self.done_flag[agent_id] = True

    def render(self, figure_scaling: float = 4.0):
        """Plot the agent's path."""
        colors = cm.rainbow(np.linspace(0, 1, len(self.agents_id)))
        i = 0
        for agent_id in self.agents_id:
            color = colors[i]
            i += 1
            for j in range(len(self.path_x[agent_id])):
                plt.plot(self.path_x[agent_id][j], self.path_y[agent_id][j], color=color)

        # plot the pins
        i = 0
        for net in self.nets:
            color = colors[i]
            x_coords, y_coords = zip(*net)
            plt.scatter(x_coords, y_coords, color=color)
            i += 1

        # plot the macro pins
        for point in self.macros:
            x, y = point
            plt.scatter(x, y, color='black')

        # set the labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Nets')

        # set the x and y axis ticks
        plt.xticks(range(0, self.length))
        plt.yticks(range(0, self.width))

        # set the grid
        plt.grid(color='blue', linestyle='--', linewidth=0.5)

        # show the plot
        fig = plt.gcf()
        fig.set_size_inches(self.length/figure_scaling, self.width/figure_scaling)
        plt.show()

    def heatmap(self, figure_scaling: float = 4.0):
        """Generate the horizontal and vertical edge usage heat maps."""
        horizontal_edge_index = 1  # indices for the up and right edge
        vertical_edge_index = 0
        shape = self.edge_capacity.shape
        horizontal_usage = self.edge_capacity[0:shape[0]-1, :, [horizontal_edge_index]].squeeze().transpose()
        vertical_usage = self.edge_capacity[:, 0:shape[1]-1, [vertical_edge_index]].squeeze().transpose()
        ticks = np.arange(0, self.max_capacity)

        plt.figure()
        sns.heatmap(horizontal_usage, annot=True, cmap="Reds", square=True, cbar_kws={"ticks": ticks})
        plt.gca().invert_yaxis()
        fig = plt.gcf()
        fig.set_size_inches(self.length/figure_scaling, self.width/figure_scaling)
        plt.title("Horizontal Heat Map")
        plt.show()

        plt.figure()
        sns.heatmap(vertical_usage, annot=True, cmap="Reds", square=True, cbar_kws={"ticks": ticks})
        plt.gca().invert_yaxis()
        fig = plt.gcf()
        fig.set_size_inches(self.length/figure_scaling, self.width/figure_scaling)
        plt.title("Vertical Heat Map")
        plt.show()

    def reset(self):
        self.reset_pin_counters()
        self.step_counter = 0
        self.change_pin_flag = self.reset_flags(self.change_pin_flag)
        self.done_flag = self.reset_flags(self.done_flag)
        self.done_flag["__all__"] = False
        self.edge_capacity = self.initial_capacity.copy()
        self.path_x = self.generate_path(self.nets)
        self.path_y = self.generate_path(self.nets)
        for agent_id in self.agents_id:
            self.update_positions(agent_id)
            self.update_path(agent_id)
        for agent_id in self.agents_id:
            individual_state = np.concatenate([
                self.agent_position[agent_id],
                self.goal_position[agent_id],
                self.radar(self.agent_position[agent_id])
                ])
            self.state[agent_id] = individual_state

        return self.state

    def random_action(self):
        action = {}
        for agent_id in self.agents_id:
            action[agent_id] = random.randrange(0, 4)

        return action


class MultiEnv(MultiAgentEnv):
    """A wrapper of multiple environments to aid for training for generalizability"""
    def __init__(self, benchmarks: list, default_env: int = 0):
        # Store each benchmark as a separate RtGridEnv instance
        self.envs = [RtGridEnv(**benchmark) for benchmark in benchmarks]
        self.current_env = self.envs[default_env]

        self.action_space = self.current_env.action_space

        max_length = max(item['length'] for item in benchmarks)
        max_width = max(item['width'] for item in benchmarks)
        self.observation_space = MultiDiscrete([max_length, max_width, max_length, max_width, 2, 2, 2, 2])

    def switch_benchmark(self, benchmark_idx):
        # Change the current environment to a new benchmark
        self.current_env = self.envs[benchmark_idx]

    def step(self, action_dict):
        return self.current_env.step(action_dict)

    def reset(self):
        return self.current_env.reset()


class SwitchBenchmarkCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, episode, **kwargs):
        # When an episode ends, switch to the next benchmark.
        # Here we're just rotating benchmarks for simplicity.
        env = base_env.envs[0]
        current_benchmark_idx = env.envs.index(env.current_env)
        next_benchmark_idx = (current_benchmark_idx + 1) % len(env.envs)
        env.switch_benchmark(next_benchmark_idx)


class PrintEnvCallback(DefaultCallbacks):
    def on_episode_created(self, base_env, **kwargs):
        print(base_env.envs[0].current_env.nets)
