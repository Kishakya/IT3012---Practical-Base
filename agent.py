# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)

# Step 1.2: Implementing BFS, DFS, and UCS
from collections import deque
import heapq
import itertools


class SearchAgent:
    """Uninformed search strategies (BFS, DFS, UCS) over the grid.

    All three share the same successor generation and path reconstruction;
    only the Frontier data structure differs, which is what actually
    distinguishes the three algorithms.
    """

    def __init__(self, grid_size, walls):
        self.width, self.height = grid_size
        self.walls = set(walls)
        self.plan = []
        self.active_algo = 'BFS'

    def get_successors(self, state):
        x, y = state
        moves = [
            ('Up', (x, y + 1)),
            ('Down', (x, y - 1)),
            ('Left', (x - 1, y)),
            ('Right', (x + 1, y)),
        ]
        successors = []
        for action, (nx, ny) in moves:
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.walls:
                successors.append((action, (nx, ny), 1))
        return successors

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            agent_pos = tuple(percept['agent_pos'])
            food = percept['all_food']
            if not food:
                return 'Stop'

            goal = min(food, key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1]))

            if self.active_algo == 'DFS':
                path = self.dfs_search(agent_pos, goal)
            elif self.active_algo == 'UCS':
                path = self.ucs_search(agent_pos, goal)
            else:
                path = self.bfs_search(agent_pos, goal)

            self.plan = path if path else []

        if not self.plan:
            return 'Stop'

        return self.plan.pop(0)

    def reconstruct_path(self, came_from, start, goal):
        actions = []
        state = goal
        while state != start:
            prev_state, action = came_from[state]
            actions.append(action)
            state = prev_state
        actions.reverse()
        return actions

    def bfs_search(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        if start == goal:
            return []

        frontier = deque([start])
        reached = {start}
        came_from = {}

        while frontier:
            state = frontier.popleft()
            for action, next_state, _ in self.get_successors(state):
                if next_state not in reached:
                    reached.add(next_state)
                    came_from[next_state] = (state, action)
                    if next_state == goal:
                        return self.reconstruct_path(came_from, start, goal)
                    frontier.append(next_state)
        return None

    def dfs_search(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        if start == goal:
            return []

        frontier = [start]
        reached = {start}
        came_from = {}

        while frontier:
            state = frontier.pop()
            if state == goal:
                return self.reconstruct_path(came_from, start, goal)
            for action, next_state, _ in self.get_successors(state):
                if next_state not in reached:
                    reached.add(next_state)
                    came_from[next_state] = (state, action)
                    frontier.append(next_state)
        return None

    def ucs_search(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        if start == goal:
            return []

        tie_breaker = itertools.count()
        frontier = [(0, next(tie_breaker), start)]
        reached = {start: 0}
        came_from = {}

        while frontier:
            cost, _, state = heapq.heappop(frontier)
            if state == goal:
                return self.reconstruct_path(came_from, start, goal)
            if cost > reached.get(state, float('inf')):
                continue
            for action, next_state, step_cost in self.get_successors(state):
                new_cost = cost + step_cost
                if next_state not in reached or new_cost < reached[next_state]:
                    reached[next_state] = new_cost
                    came_from[next_state] = (state, action)
                    heapq.heappush(frontier, (new_cost, next(tie_breaker), next_state))
        return None

