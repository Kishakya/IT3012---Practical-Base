# agent.py
import random
import math
import heapq
from collections import deque

from logic_engine import KnowledgeBase


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)

class SearchAgent:
    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo  # Options: 'BFS', 'DFS', 'UCS'

        self.kb = KnowledgeBase()
        self.kb.tell_rule(["TargetVisible", "HasDust"], "SafeToEngage")
        self.kb.tell_rule(["SafeToEngage", "BloodseekerMissing"], "Retreat")

    def get_neighbors(self, state, grid_size, walls):
        x, y = state
        width, height = grid_size
        actions = [
            ('UP', (x, y + 1)),
            ('DOWN', (x, y - 1)),
            ('LEFT', (x - 1, y)),
            ('RIGHT', (x + 1, y))
        ]
        valid_neighbors = []
        for action, (nx, ny) in actions:
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                valid_neighbors.append((action, (nx, ny)))
        return valid_neighbors

    def bfs_search(self, start, target, grid_size, walls):
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            state, path = frontier.popleft()
            if state == target:
                return path

            for action, neighbor in self.get_neighbors(state, grid_size, walls):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append((neighbor, path + [action]))
        return []

    def dfs_search(self, start, target, grid_size, walls):
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            state, path = frontier.pop()
            if state == target:
                return path

            for action, neighbor in self.get_neighbors(state, grid_size, walls):
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append((neighbor, path + [action]))
        return []

    def ucs_search(self, start, target, grid_size, walls):
        counter = 0
        frontier = [(0, counter, start, [])]
        reached = {start: 0}

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if state == target:
                return path

            for action, neighbor in self.get_neighbors(state, grid_size, walls):
                new_cost = cost + 1
                if neighbor not in reached or new_cost < reached[neighbor]:
                    reached[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, neighbor, path + [action]))
        return []

    def manhattan_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan',
                      food_positions=None, opponent_positions=None):
        heuristic = self.manhattan_distance if heuristic_type == 'manhattan' else self.euclidean_distance

        food_positions = set(food_positions) if food_positions else set()
        opponent_positions = {tuple(pos) for pos in opponent_positions} if opponent_positions else set()

        counter = 0
        start_h = heuristic(start_pos, goal_pos)
        frontier = [(start_h, 0, counter, start_pos, [])]
        reached_states = set()

        while frontier:
            f_cost, g_cost, _, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for action, neighbor in self.get_neighbors(current_pos, grid_size, walls):
                if neighbor not in reached_states:
                    # Consult the Knowledge Base before treating this neighbor as feasible.
                    self.kb.clear_facts()
                    if neighbor in food_positions:
                        self.kb.tell_fact("TargetVisible")
                    else:
                        self.kb.tell_fact("HasDust")
                    if neighbor not in opponent_positions:
                        self.kb.tell_fact("BloodseekerMissing")
                    self.kb.forward_chain()

                    if "Retreat" in self.kb.facts:
                        continue  # Infeasible: skip this neighbor even though it's physically reachable.

                    g_new = g_cost + 1
                    h_new = heuristic(neighbor, goal_pos)
                    f_new = g_new + h_new
                    counter += 1
                    heapq.heappush(frontier, (f_new, g_new, counter, neighbor, path_taken + [action]))
        return []

    def sense_and_act(self, percept):
        if not self.plan:
            start = tuple(percept['agent_pos'])
            all_food = percept['all_food']
            if not all_food:
                return 'STOP'

            # Select nearest food using Manhattan distance
            target = min(all_food, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))

            grid_size = percept['grid_size']
            walls = percept['walls']

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, target, grid_size, walls)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, target, grid_size, walls)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, target, grid_size, walls)
            elif self.active_algo == 'AStar':
                self.plan = self.astar_search(
                    start, target, walls, grid_size, heuristic_type='manhattan',
                    food_positions=all_food, opponent_positions=percept.get('opponent_positions', [])
                )

        return self.plan.pop(0) if self.plan else 'STOP'


if __name__ == "__main__":
    agent = SearchAgent()
    start_pos, goal_pos = (0, 0), (3, 4)
    print(f"Manhattan distance {start_pos} -> {goal_pos}: {agent.manhattan_distance(start_pos, goal_pos)}")
    print(f"Euclidean distance {start_pos} -> {goal_pos}: {agent.euclidean_distance(start_pos, goal_pos)}")
