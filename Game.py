"""
    File name: Game.py
    Author: Ali Khaleqi Yekta [@YektaDev] - Me@Yekta.Dev
    Date created: 2023-03-27
    Python Version: 3.11.0

    The following script is an implementation of a basic game in Pac-Man style. It demonstrates the use of classic AI as
    the controller of the agent (Pac-Man) of the game, and it tries to find the shortest path to the food (the goal) by
    only knowing its own sequence of actions and positions; meaning it can't "see" the surroundings unless it tries to
    move to them.

    This script is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General
    Public License (GNU AGPL) License version 3 as published by the Free Software Foundation.

    This script is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
    details.
"""

import math
import random
import time
from abc import abstractmethod
from enum import Enum

import pygame

log = True
gui = True
random_input = True
random_gui_repeat_after_done = True
random_input_field_width = 30
random_input_field_height = 15
gui_block_size = 30

pos_history = []
action_history = []


def lg(*text):
    if log:
        print(*text)


class AgentAlgorithm(Enum):
    # A home-made algorithm!
    OLD: int = 1
    # Depth-first search algorithm
    DFS: int = 2
    # Depth-first search algorithm + The ability to see neighbour blocks.
    DFS_FORESEEN: int = 3
    # Uniform-cost search algorithm
    UCS: int = 4


class AgentPlaces:
    def __init__(self):
        self.current_y = None
        self.current_x = None

        self.above_y = None
        self.above_x = None
        self.below_y = None
        self.below_x = None
        self.left_y = None
        self.left_x = None
        self.right_y = None
        self.right_x = None
        self.above = None
        self.below = None
        self.left = None
        self.right = None

    def update_pos(self, current_y, current_x):
        self.current_y = current_y
        self.current_x = current_x

        self.above_y = current_y - 1
        self.above_x = current_x
        self.below_y = current_y + 1
        self.below_x = current_x
        self.left_y = current_y
        self.left_x = current_x - 1
        self.right_y = current_y
        self.right_x = current_x + 1

    def update_content(self, above, below, left, right):
        self.above = above
        self.below = below
        self.left = left
        self.right = right


class Agent:
    def __init__(self,
                 y: int,
                 x: int,
                 field_width,
                 field_height,
                 ucs_right_cost: int,
                 ucs_bottom_cost: int,
                 ucs_left_cost: int,
                 ucs_top_cost: int,
                 algorithm: AgentAlgorithm
                 ):
        self.places = AgentPlaces()
        self.has_food = False
        self.known_map = []
        self.known_map.append(['*'] * field_width)
        for i in range(1, field_height - 1):
            self.known_map.append(['*'] + [None] * (field_width - 2) + ['*'])
        self.known_map[y][x] = "-"
        self.known_map.append(['*'] * field_width)
        pos_history.append((y, x))
        self.algorithm: AgentAlgorithm = algorithm
        self.brain: AgentBrain
        match self.algorithm:
            case AgentAlgorithm.OLD:
                self.brain = AgentOldBrain(
                    root_y=y,
                    root_x=x,
                    ucs_right_cost=ucs_right_cost,
                    ucs_bottom_cost=ucs_bottom_cost,
                    ucs_left_cost=ucs_left_cost,
                    ucs_top_cost=ucs_top_cost
                )
            case AgentAlgorithm.DFS_FORESEEN:
                self.brain = AgentDfsForeseenBrain(
                    root_y=y,
                    root_x=x,
                    ucs_right_cost=ucs_right_cost,
                    ucs_bottom_cost=ucs_bottom_cost,
                    ucs_left_cost=ucs_left_cost,
                    ucs_top_cost=ucs_top_cost
                )
            case AgentAlgorithm.DFS:
                self.brain = AgentDfsBrain(
                    root_y=y,
                    root_x=x,
                    ucs_right_cost=ucs_right_cost,
                    ucs_bottom_cost=ucs_bottom_cost,
                    ucs_left_cost=ucs_left_cost,
                    ucs_top_cost=ucs_top_cost
                )
            case AgentAlgorithm.UCS:
                self.brain = AgentUcsBrain(
                    root_y=y,
                    root_x=x,
                    ucs_right_cost=ucs_right_cost,
                    ucs_bottom_cost=ucs_bottom_cost,
                    ucs_left_cost=ucs_left_cost,
                    ucs_top_cost=ucs_top_cost
                )
            case _:
                raise ValueError("Invalid Brain!")

        lg("Agent initialized at position: " + str((x, y)))

    def act_based_on_perception(self, perception):
        """
        Enables the agent to act based on the perception of the environment
        """

        if perception is None:
            # First action
            random_action = random.choice(["up", "down", "left", "right"])
            action_history.append(random_action)
            lg("Agent had no perception whatsoever, so it moved randomly to: " + random_action)
            return random_action

        prev_action = action_history[-1]
        prev_y = pos_history[-1][0]
        prev_x = pos_history[-1][1]
        new_y = perception[0]
        new_x = perception[1]
        pos_history.append((new_y, new_x))
        lg("Agent was in", str((prev_x, prev_y)), "and is now in", str((new_x, new_y)))

        # Check if the agent has eaten the food
        has_food = perception[2]
        if has_food:
            self.known_map[new_y][new_x] = "f"
            if log:
                print("Agent ate the food!")
            self.print_result()
            return None
        elif log:
            print("Agent's Memory Dump:")
            print('\n'.join([''.join(map(stringify, row)) for row in self.known_map]))

        if prev_y == new_y and prev_x == new_x:
            # Agent couldn't move. It must be a wall!
            lg("Agent couldn't move, because the previous action lead to a wall. Wall marked.")
            if prev_action == "up":
                self.known_map[new_y - 1][new_x] = "*"
            elif prev_action == "down":
                self.known_map[new_y + 1][new_x] = "*"
            elif prev_action == "left":
                self.known_map[new_y][new_x - 1] = "*"
            elif prev_action == "right":
                self.known_map[new_y][new_x + 1] = "*"
        else:
            # Agent moved. It must be a path!
            lg("Agent moved. Current position marked as free.")
            self.known_map[new_y][new_x] = "-"

        # Decide the next action based on the known map
        action = self.decide_next_action()
        action_history.append(action)
        lg("Agent decided to take action: " + action)

        return action

    def decide_next_action(self):
        """
        :return the next action that the agent will take
        """
        result = "repeat"
        r = 0
        while result == "repeat":
            r += 1
            if r > 4:
                exit(0)
            current_y = pos_history[-1][0]
            current_x = pos_history[-1][1]
            self.places.update_pos(current_y=current_y, current_x=current_x)
            self.places.update_content(
                above=self.known_map[self.places.above_y][self.places.above_x],
                below=self.known_map[self.places.below_y][self.places.below_x],
                left=self.known_map[self.places.left_y][self.places.left_x],
                right=self.known_map[self.places.right_y][self.places.right_x]
            )
            result = self.brain.next_action(self)
            if result == "repeat":
                lg("Agent's brain requested another thinking iteration to decide the next action.")
        return result

    def print_result(self):
        """
        Print the result of the whole simulation
        """
        print_title("Agent's initial position")
        print("(x:", str(pos_history[0][1]) + ", y:", str(pos_history[0][0]) + ")")

        print_title("Food's position")
        print("(x:", str(pos_history[-1][1]) + ", y:", str(pos_history[-1][0]) + ")")

        print_title("Agent's total moves")
        print(len(action_history))

        print_title("Agent's history of positions and taken actions")
        positions_and_actions = str(pos_history[0]).title()
        for i in range(len(action_history)):
            positions_and_actions += " " + Color.CYAN + action_history[i].title().rjust(6)
            positions_and_actions += Color.RED + " -> " + Color.END
            if i % 3 == 2:
                positions_and_actions += "\n"
            positions_and_actions += str(pos_history[i + 1])
        print("\n" + positions_and_actions)

        print_title("Agent's Final Memory Dump")
        print("\n" + '\n'.join([''.join(map(stringify, row)) for row in self.known_map]))

        if not log:
            print(
                Color.YELLOW + Color.UNDERLINE + ">> To see the detailed step-by-step logs in future executions, set " +
                Color.BOLD + Color.RED + "log" + Color.END +
                Color.YELLOW + Color.UNDERLINE + " to " +
                Color.BOLD + Color.PINK + "True" + Color.END +
                Color.YELLOW + Color.UNDERLINE + "." + Color.END
            )


class AgentBrain:
    def __init__(self, root_y, root_x, ucs_right_cost, ucs_bottom_cost, ucs_left_cost, ucs_top_cost):
        self.root_y = root_y
        self.root_x = root_x
        self.ucs_right_cost = ucs_right_cost
        self.ucs_bottom_cost = ucs_bottom_cost
        self.ucs_left_cost = ucs_left_cost
        self.ucs_top_cost = ucs_top_cost

    @abstractmethod
    def next_action(self, agent) -> str:
        pass


class AgentOldBrain(AgentBrain):
    def next_action(self, agent) -> str:
        new_actions = get_new_actions_possible(agent.places)

        # If there are any new actions, then choose one of them randomly.
        if len(new_actions) > 0:
            lg("Agent found that these actions lead to a new position that's not traveled before:", str(new_actions))
            random_action = random.choice(new_actions)
            lg("Agent chose to take one of them randomly:", random_action)
            return random_action

        # If there are no new actions, then choose one of the available actions that is traveled the least.
        lg("Agent found that all currently-possible actions lead to a position that's already traveled before.")
        above_pos_travel_count = pos_history.count((agent.places.above_y, agent.places.above_x))
        below_pos_travel_count = pos_history.count((agent.places.below_y, agent.places.below_x))
        left_pos_travel_count = pos_history.count((agent.places.left_y, agent.places.left_x))
        right_pos_travel_count = pos_history.count((agent.places.right_y, agent.places.right_x))

        # If it's known but not traveled, it's a wall
        if above_pos_travel_count == 0:
            above_pos_travel_count = 999999999999999
        if below_pos_travel_count == 0:
            below_pos_travel_count = 999999999999999
        if left_pos_travel_count == 0:
            left_pos_travel_count = 999999999999999
        if right_pos_travel_count == 0:
            right_pos_travel_count = 999999999999999

        lg(
            "Travel counts for each direction (999999999999999 indicates a wall):",
            "Up:", above_pos_travel_count,
            "Down:", below_pos_travel_count,
            "Left:", left_pos_travel_count,
            "Right:", right_pos_travel_count,
        )

        travels = sorted(
            [
                ("up", above_pos_travel_count),
                ("down", below_pos_travel_count),
                ("left", left_pos_travel_count),
                ("right", right_pos_travel_count),
            ],
            key=lambda item: item[1],
        )

        # Find all the actions that lead to the least traveled position
        mins = [item for item in travels if item[1] == travels[0][1]]
        # Choose the one which it's last occurrence has been called sooner than the others
        # to avoid going where it's been recently. It's just a design choice.
        for action in action_history.__reversed__():
            if len(mins) <= 1:
                break
            if action in mins:
                mins.remove(action)

        lg("Agent chose to take (one of) the action(s) that leads to the least traveled position:", mins[0][0])
        return mins[0][0]


class AgentDfsBrain(AgentBrain):
    def next_action(self, agent) -> str:
        pass


class AgentDfsForeseenBrain(AgentBrain):
    def next_action(self, agent) -> str:
        pass


class AgentUcsBrain(AgentBrain):
    def __init__(self, root_x, root_y, ucs_right_cost, ucs_bottom_cost, ucs_left_cost, ucs_top_cost):
        super().__init__(root_y=root_y, root_x=root_x,
                         ucs_right_cost=ucs_right_cost, ucs_bottom_cost=ucs_bottom_cost,
                         ucs_left_cost=ucs_left_cost, ucs_top_cost=ucs_top_cost)
        # True: It's discovering new nodes
        # False: It's traveling to a node that's considered to have the least cost.
        self.explore_mode = True
        # When explore_mode = False:
        self.go_towards_root = True
        self.actions_to_go_from_root_to_target = []
        self.go_to_target = None

        self.here = None
        self.prev_pos = None

        # Actions that need to be done from the starting point to get to the point of the given position as key.
        self.start_to_pos_actions: dict = {
            (self.root_y, self.root_x): [],
        }

    def next_action(self, agent) -> str:
        self.here = (agent.places.current_y, agent.places.current_x)

        if self.here not in self.start_to_pos_actions:
            if self.prev_pos is not None:
                self.start_to_pos_actions[self.here] = self.start_to_pos_actions.get(self.prev_pos) + [
                    action_history[-1]]
            else:
                self.start_to_pos_actions[self.here] = [action_history[-1]]

        if self.explore_mode:
            # Explore new nodes. Falling into this branch implies we're definitely at a leaf in our visited tree.

            # The actual UCS
            self.update_costs_and_paths_to_known_neighbours(agent.places, agent.known_map)
            new_actions = get_new_actions_possible(agent.places)
            if len(new_actions) == 0:  # We're at a leaf with no unseen stuff
                self.explore_mode = False
                self.go_to_target = self.get_least_cost_known_node()  # An Important Part!
                lg(self.go_to_target, 'was chosen to be the next target, since it has the lowest cost.')

                self.prev_pos = self.here
                return "repeat"

            self.prev_pos = self.here
            return new_actions.pop(0)
        else:
            # Just continue traveling.
            # Implementation Note:
            # This is raw UCS. For simplicity, in order for the agent to go from visited node A to visited node B, it
            # ALWAYS first travels from A to the initial root, and then from the root to B. This, of course, can be
            # optimized to set the agent to only travel to the nearest root that connects the two nodes, but this is out
            # of the scope for this implementation.
            if self.go_towards_root:
                # Go one more step towards the root = reverse the last action to get to this node
                here_actions = self.start_to_pos_actions.get(self.here)
                if len(here_actions) < 1:
                    # This is the root
                    self.go_towards_root = False
                    self.actions_to_go_from_root_to_target = list(self.start_to_pos_actions.get(self.go_to_target))
                    self.prev_pos = self.here

                    return "repeat"
                lg(f'Agent wants to go towards the root so that it can find its way to {self.go_to_target} afterwards.')
                last_action_to_get_here = here_actions[-1]
                lg('so it repeats the reverse of last action that caused it to reach', self.here)
                self.prev_pos = self.here
                return reverse_action(last_action_to_get_here)
            else:
                if len(self.actions_to_go_from_root_to_target) == 0:
                    # We reached the leaf target
                    self.explore_mode = True
                    self.go_towards_root = True
                    self.go_to_target = None
                    self.prev_pos = self.here
                    return "repeat"  # Run one more time to explore
                # Follow the sequence to get to the target
                self.prev_pos = self.here
                return self.actions_to_go_from_root_to_target.pop(0)

    def get_least_cost_known_node(self):
        pos_cost_list = []
        for pos, actions in self.start_to_pos_actions.items():
            pos_cost_list.append((pos, self.get_ucs_cost_of_actions(actions)))
        min_pos_cost = min(pos_cost_list, key=lambda t: t[1])
        return min_pos_cost[0]

    def update_costs_and_paths_to_known_neighbours(self, places, known_map):
        def current_cost_of(pos):
            actions_from_root = self.start_to_pos_actions.get(pos)
            return self.get_ucs_cost_of_actions(actions_from_root)

        def has_any_undiscovered_way(pos):
            (y, x) = pos
            has_u_down = y < len(known_map) - 1 and known_map[y + 1][x] == '?'
            has_u_right = x < len(known_map) - 1 and known_map[y][x + 1] == '?'
            has_u_up = y > 0 and known_map[y - 1][x] == '?'
            has_u_left = x > 0 and known_map[y][x - 1] == '?'
            return has_u_down or has_u_up or has_u_right or has_u_left

        heavy = ['up', 'down', 'left', 'right'] * 500

        if places.right == "-" or places.right == "f":  # Has a known right
            if len(self.start_to_pos_actions.get(self.here)) > 0:
                right_pos = (places.right_y, places.right_x)
                current_right_cost = current_cost_of(right_pos)
                here_to_right_actions = self.start_to_pos_actions.get(self.here) + ["right"]
                here_to_right_cost = self.get_ucs_cost_of_actions(here_to_right_actions)
                lg('Cost of root to right neighbour:', current_right_cost, 'Cost of here to right neighbour:',
                   here_to_right_cost)
                if here_to_right_cost < current_right_cost:
                    self.start_to_pos_actions[right_pos] = here_to_right_actions
                if not has_any_undiscovered_way(right_pos):
                    self.start_to_pos_actions[right_pos] = heavy

        if places.below == "-" or places.below == "f":  # Has a known bottom
            if len(self.start_to_pos_actions.get(self.here)) > 0:
                bottom_pos = (places.below_y, places.below_x)
                current_bottom_cost = current_cost_of(bottom_pos)
                print(type(self.start_to_pos_actions))
                here_to_bottom_actions = self.start_to_pos_actions.get(self.here) + ["down"]
                here_to_bottom_cost = self.get_ucs_cost_of_actions(here_to_bottom_actions)
                lg('Cost of root to below neighbour:', current_bottom_cost, 'Cost of here to below neighbour:',
                   here_to_bottom_cost)
                if here_to_bottom_cost < current_bottom_cost:
                    self.start_to_pos_actions[bottom_pos] = here_to_bottom_actions
                if not has_any_undiscovered_way(bottom_pos):
                    self.start_to_pos_actions[bottom_pos] = heavy

        if places.left == "-" or places.left == "f":  # Has a known left
            if len(self.start_to_pos_actions.get(self.here)) > 0:
                left_pos = (places.left_y, places.left_x)
                current_left_cost = current_cost_of(left_pos)
                here_to_left_actions = self.start_to_pos_actions.get(self.here) + ["left"]
                here_to_left_cost = self.get_ucs_cost_of_actions(here_to_left_actions)
                lg('Cost of root to left neighbour:', current_left_cost, 'Cost of here to left neighbour:',
                   here_to_left_cost)
                if here_to_left_cost < current_left_cost:
                    self.start_to_pos_actions[left_pos] = here_to_left_actions
                if not has_any_undiscovered_way(left_pos):
                    self.start_to_pos_actions[left_pos] = heavy

        if places.above == "-" or places.above == "f":  # Has a known top
            if len(self.start_to_pos_actions.get(self.here)) > 0:
                top_pos = (places.above_y, places.above_x)
                current_top_cost = current_cost_of(top_pos)
                here_to_top_actions = self.start_to_pos_actions.get(self.here) + ["up"]
                here_to_top_cost = self.get_ucs_cost_of_actions(here_to_top_actions)
                lg('Cost of root to above neighbour:', current_top_cost, 'Cost of here to left neighbour:',
                   here_to_top_cost)
                if here_to_top_cost < current_top_cost:
                    self.start_to_pos_actions[top_pos] = here_to_top_actions
                if not has_any_undiscovered_way(top_pos):
                    self.start_to_pos_actions[top_pos] = heavy

    def get_ucs_cost_of_actions(self, actions) -> int:
        total = 0
        for action in actions:
            match action:
                case 'up':
                    total += self.ucs_top_cost
                case 'down':
                    total += self.ucs_bottom_cost
                case 'left':
                    total += self.ucs_left_cost
                case 'right':
                    total += self.ucs_right_cost
                case _:
                    raise ValueError("Invalid Action: " + action)
        return total


def is_wall(known_map, y, x):
    out_of_bounds = y < 1 or x < 1
    out_of_bounds = out_of_bounds or (y >= len(known_map))
    out_of_bounds = out_of_bounds or (x >= len(known_map[0]))
    return out_of_bounds or known_map[y][x] == '*'


def reverse_action(action: str):
    match action:
        case 'up':
            return 'down'
        case 'down':
            return 'up'
        case 'left':
            return 'right'
        case 'right':
            return 'left'
        case _:
            raise ValueError("Invalid Action: " + action)


def get_new_actions_possible(places: AgentPlaces):
    # Check the four directions for available paths (not yet visited / not walls)
    # Precedence: Right, Down, Left, Up
    available_actions = []
    if places.right != "*":
        available_actions.append("right")
    if places.below != "*":
        available_actions.append("down")
    if places.left != "*":
        available_actions.append("left")
    if places.above != "*":
        available_actions.append("up")

    if len(available_actions) == 0:
        print("ERROR: No available actions found. This means the agent is in a 1x1 box surrounded by walls.")
        return None

    new_actions = []
    for action in available_actions:
        if action == "up" and places.above != "-":
            new_actions.append(action)
        elif action == "down" and places.below != "-":
            new_actions.append(action)
        elif action == "left" and places.left != "-":
            new_actions.append(action)
        elif action == "right" and places.right != "-":
            new_actions.append(action)
    return new_actions


class Environment:
    def __init__(self, food_pos, walls):
        self.food_pos = food_pos
        self.walls = walls
        self.width = len(walls[0])
        self.height = len(walls)

    def percept_environment(self, agent_y, agent_x, action):
        """
        :return the perception of the agent from the environment after taking an action, which is an array consisting of
        three elements: [agent_x, agent_y, has_food]
        """

        # Update the position of the agent
        (new_agent_y, new_agent_x) = self.get_new_pos_after_action(agent_y, agent_x, action)

        # Check if the agent has eaten the food
        has_food = new_agent_y == self.food_pos[0] and new_agent_x == self.food_pos[1]

        return [new_agent_y, new_agent_x, has_food]

    def get_new_pos_after_action(self, agent_y, agent_x, action):
        cant_move = False

        cant_move = cant_move or (action == "up" and agent_y < 1)
        cant_move = cant_move or (action == "down" and agent_y > self.height - 2)
        cant_move = cant_move or (action == "left" and agent_x < 1)
        cant_move = cant_move or (action == "right" and agent_x > self.width - 2)

        cant_move = cant_move or (action == "up" and self.walls[agent_y - 1][agent_x] is True)
        cant_move = cant_move or (action == "down" and self.walls[agent_y + 1][agent_x] is True)
        cant_move = cant_move or (action == "left" and self.walls[agent_y][agent_x - 1] is True)
        cant_move = cant_move or (action == "right" and self.walls[agent_y][agent_x + 1] is True)

        if cant_move:
            return agent_y, agent_x

        if action == "up":
            return agent_y - 1, agent_x
        if action == "down":
            return agent_y + 1, agent_x
        if action == "left":
            return agent_y, agent_x - 1
        if action == "right":
            return agent_y, agent_x + 1


class Color:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_title(title):
    print(Color.BOLD + Color.GREEN + title + Color.PINK + ": " + Color.END, end="")


def stringify(string):
    if string is None:
        return "?"
    if string == "*":
        return Color.RED + string + Color.END
    if string == "-":
        return Color.BLUE + string + Color.END
    if string == "f":
        return Color.BOLD + Color.YELLOW + string + Color.END
    return string


class Game:
    def __init__(self, game_agent: Agent, game_environment: Environment):
        self.game_agent = game_agent
        self.game_environment = game_environment

    def run(self):
        print(Color.BOLD + Color.UNDERLINE + "** PacManAI By Ali Khaleqi Yekta [Me@Yekta.Dev] **" + Color.END)
        print(Color.BOLD + Color.PINK +
              f'>> Simulation Started with the {str(self.game_agent.algorithm.name)} algorithm...'
              + Color.END)
        agent_action = self.game_agent.act_based_on_perception(None)

        screen = None  # Dummy
        clock = None  # Dummy
        if gui:
            pygame.init()
            clock = pygame.time.Clock()
            screen_size = (
                len(self.game_environment.walls[0]) * gui_block_size, len(self.game_environment.walls) * gui_block_size)
            screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("PacManAI - By Ali Khaleqi Yekta [Me@Yekta.Dev]")
        while True:
            if agent_action is None:
                time.sleep(1)
                return

            if gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit(0)

            agent_perception = self.game_environment.percept_environment(
                agent_y=pos_history[-1][0],
                agent_x=pos_history[-1][1],
                action=agent_action,
            )
            agent_action = self.game_agent.act_based_on_perception(agent_perception)

            if gui:
                (food_y, food_x) = self.game_environment.food_pos
                render(screen, self.game_environment.walls, agent_perception[1], agent_perception[0], food_x, food_y)
                for i in range(10):
                    clock.tick(100)
            else:
                time.sleep(.125)


class FileGameDataLoader:
    def __init__(self, file_path, algorithm: AgentAlgorithm):
        self.file_path = file_path
        self.env_width = -1
        self.env_height = -1
        self.agent_x = -1
        self.agent_y = -1

        data = list(filter(lambda item: item.strip() or None, self.load_as_list()))
        (self.env_height, self.env_width) = data[0].split(',')
        (self.agent_x, self.agent_y) = data[1].split(',')
        (self.ucs_right_cost, self.ucs_bottom_cost, self.ucs_left_cost, self.ucs_top_cost) = data[2].split(',')
        self.env_width = int(self.env_width)
        self.env_height = int(self.env_height)
        self.agent_x = int(self.agent_x)
        self.agent_y = int(self.agent_y)
        self.ucs_right_cost = int(self.ucs_right_cost)
        self.ucs_bottom_cost = int(self.ucs_bottom_cost)
        self.ucs_left_cost = int(self.ucs_left_cost)
        self.ucs_top_cost = int(self.ucs_top_cost)

        env_text = data[3:]
        env_walls_bool = []
        for line in env_text:
            env_walls_bool.append([True if char == "*" else False for char in line])

        food_x = -1  # Dummy
        food_y = -1  # Dummy
        for line in env_text:
            if "f" in line:
                food_x = line.index("f")
                food_y = env_text.index(line)
                break

        self.environment = Environment(food_pos=(food_y, food_x), walls=env_walls_bool)
        self.agent = Agent(
            y=self.agent_y,
            x=self.agent_x,
            field_width=self.env_width,
            field_height=self.env_height,
            algorithm=algorithm,
            ucs_right_cost=self.ucs_right_cost,
            ucs_bottom_cost=self.ucs_bottom_cost,
            ucs_left_cost=self.ucs_left_cost,
            ucs_top_cost=self.ucs_top_cost,
        )
        if log:
            print_title("Loaded Input Data")
            ucs_data: str = ''
            if algorithm is AgentAlgorithm.UCS:
                ucs_data += "UCS direction weights:\n"
                ucs_data += f'  Right: {str(self.ucs_right_cost)}\n'
                ucs_data += f'  Bottom: {str(self.ucs_bottom_cost)}\n'
                ucs_data += f'  Left: {str(self.ucs_left_cost)}\n'
                ucs_data += f'  Top: {str(self.ucs_top_cost)}'
            print(
                "width: " + str(self.env_width),
                "height: " + str(self.env_height),
                "agent_x: " + str(self.agent_x),
                "agent_y: " + str(self.agent_y),
                "food_y: " + str(food_y),
                "food_x: " + str(food_x),
                ucs_data
            )

    def load_as_list(self):
        with open(self.file_path, "r") as file:
            data = file.readlines()
        return data


class RandomGameDataGenerator:
    def __init__(self, width, height, algorithm: AgentAlgorithm):
        self.width = width
        self.height = height
        self.food_y = random.randint(1, height - 2)
        self.food_x = random.randint(1, width - 2)
        self.agent_y = -1  # Dummy
        self.agent_x = -1  # Dummy
        while True:
            self.agent_y = random.randint(1, height - 2)
            self.agent_x = random.randint(1, width - 2)
            if self.agent_y != self.food_y and self.agent_x != self.food_x:
                break

        self.environment = Environment(food_pos=(self.food_y, self.food_x), walls=self.generate_random_walls())
        ucs_right_cost = random.randint(1, 5)
        ucs_bottom_cost = random.randint(1, 5)
        ucs_left_cost = random.randint(1, 5)
        ucs_top_cost = random.randint(1, 5)
        if algorithm is AgentAlgorithm.UCS:
            print("Randomly generated UCS direction weights:")
            print(f'  Right: {ucs_right_cost}')
            print(f'  Bottom: {ucs_bottom_cost}')
            print(f'  Left: {ucs_left_cost}')
            print(f'  Top: {ucs_top_cost}')

        self.agent = Agent(
            y=self.agent_y,
            x=self.agent_x,
            field_width=width,
            field_height=height,
            algorithm=algorithm,
            ucs_right_cost=ucs_right_cost,
            ucs_bottom_cost=ucs_bottom_cost,
            ucs_left_cost=ucs_left_cost,
            ucs_top_cost=ucs_top_cost,
        )

    def generate_random_walls(self):
        walls = [[True] * self.width]

        inner_maze = self.generate_inner_maze(self.width - 2, self.height - 2)
        for i in range(self.height - 2):
            walls.append([True] + inner_maze[i] + [True])

        walls.append([True] * self.width)
        return walls

    def generate_inner_maze(self, inner_width, inner_height):
        # Initialize the maze with all walls
        maze = [[True for _ in range(inner_width)] for _ in range(inner_height)]

        inner_agent_x = self.agent_x - 1
        inner_agent_y = self.agent_y - 1
        inner_food_x = self.food_x - 1
        inner_food_y = self.food_y - 1

        # Mark the player and food locations as open
        maze[inner_agent_y][inner_agent_x] = False
        maze[inner_food_y][inner_food_x] = False

        # Create a random path from the player to the food
        (x, y) = inner_agent_x, inner_agent_y
        while x != inner_food_x or y != inner_food_y:
            directions = []
            if x < inner_food_x:
                directions.append((1, 0))
            elif x > inner_food_x:
                directions.append((-1, 0))
            if y < inner_food_y:
                directions.append((0, 1))
            elif y > inner_food_y:
                directions.append((0, -1))
            (dx, dy) = random.choice(directions)
            x += dx
            y += dy
            maze[y][x] = False

        # Fill the remaining parts of the maze with walls
        for i in range(inner_height):
            for j in range(inner_width):
                if maze[i][j] is True:
                    maze[i][j] = random.random() < .45  # 45% chance of being a wall

        return maze


def render(screen, walls, player_x, player_y, food_x, food_y):
    black = (0, 0, 0)
    yellow = (255, 255, 0)
    red = (255, 0, 0)
    pink = (255, 0, 200)
    blue = (0, 0, 255)
    screen.fill(black)

    # Walls
    for i in range(len(walls)):
        for j in range(len(walls[0])):
            if walls[i][j]:
                wall_rect = pygame.Rect(j * gui_block_size, i * gui_block_size, gui_block_size, gui_block_size)
                # Background
                pygame.draw.rect(screen, blue, wall_rect)
                # Borders
                pygame.draw.line(screen, (150, 150, 150), wall_rect.topleft, wall_rect.bottomleft, 1)
                pygame.draw.line(screen, (150, 150, 150), wall_rect.topright, wall_rect.bottomright, 1)
                pygame.draw.line(screen, (100, 100, 100), wall_rect.topleft, wall_rect.topright, 2)
                pygame.draw.line(screen, (100, 100, 100), wall_rect.bottomleft, wall_rect.bottomright, 2)

    # Agent
    player_pos = (int((player_x + .5) * gui_block_size), int((player_y + .5) * gui_block_size))
    pygame.draw.circle(screen, yellow, player_pos, gui_block_size // 2)

    # Food
    food_pos = (int((food_x + .5) * gui_block_size), int((food_y + .5) * gui_block_size))
    pygame.draw.circle(screen, pink, food_pos, gui_block_size // 4)

    # Draw outer ring of food with pulsing effect
    pulse_time = None  # Dummy
    pulse_radius = None  # Dummy
    for i in range(1, 4):
        pulse_radius = int(gui_block_size * .5)
        pulse_time = pygame.time.get_ticks() / 1000
        pulse_phase = math.sin(i / 2 + pulse_time * 2 * math.pi) * .5 + .5
        pulse_size = pulse_radius + int(pulse_phase * pulse_radius)
        pygame.draw.circle(screen, (255, 100, 100), food_pos, pulse_size, 4 - i)

    # Draw the inner ring of the food with rotating effect
    for i in range(8):
        angle = i * math.pi / 4 + pulse_time * math.pi
        offset = (math.cos(angle) * pulse_radius, math.sin(angle) * pulse_radius)
        inner_pos = (int(food_pos[0] + offset[0]), int(food_pos[1] + offset[1]))
        pygame.draw.circle(screen, red, inner_pos, gui_block_size // 7)

    pygame.display.flip()


if __name__ == "__main__":
    running = True

    while running is True:
        pos_history.clear()
        action_history.clear()
        if random_input:
            print("Random Input Mode is enabled. Game will be generated randomly.")
            random_generator = \
                RandomGameDataGenerator(random_input_field_width, random_input_field_height, AgentAlgorithm.UCS)
            game = Game(random_generator.agent, random_generator.environment)
        else:
            print("Random Input Mode is disabled. Game will be loaded from input.txt.")
            file_data_loader = FileGameDataLoader("input.txt", AgentAlgorithm.OLD)
            game = Game(file_data_loader.agent, file_data_loader.environment)
        game.run()
        running = random_gui_repeat_after_done and gui and random_input

    time.sleep(2)
    exit(0)
