import heapq
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import numpy as np

# Dijkstra's Algorithm with 8-directional movement and variable edge costs
def dijkstra(grid, start, goal, cost_fn=None):
    rows, cols = len(grid), len(grid[0])
    heap = [(0, start)]
    visited = set()
    dist = {start: 0}
    parent = {start: None}

    # 8 directions with (dx, dy) and movement cost multipliers (for diagonals)
    directions = [
        (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
    ]

    while heap:
        cost, (x, y) = heapq.heappop(heap)
        if (x, y) == goal:
            path = []
            while True:
                path.append((x, y))
                if parent[(x, y)] is None:
                    break
                x, y = parent[(x, y)]
            return path[::-1], dist[goal]

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy, move_cost in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                edge_cost = cost_fn((x, y), (nx, ny)) if cost_fn else move_cost
                next_cost = cost + edge_cost
                if (nx, ny) not in dist or next_cost < dist[(nx, ny)]:
                    dist[(nx, ny)] = next_cost
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (next_cost, (nx, ny)))
    return None, float('inf')

def smooth_path(path, num_points=100):
    if len(path) < 3:
        return path
    x, y = zip(*path)
    tck, _ = splprep([x, y], s=0)
    u = np.linspace(0, 1, num_points)
    smooth_x, smooth_y = splev(u, tck)
    return list(zip(smooth_x, smooth_y))

def plot_grid(grid, path, start, goal):
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots()

    # Plot obstacles (black)
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                ax.add_patch(plt.Rectangle((j, rows - i - 1), 1, 1, color='black', label='Obstacle' if (i == 0 and j == 3) else None))

    # Plot discrete path (cyan) and smoothed path (blue)
    if path:
        for (x, y) in path:
            ax.add_patch(plt.Rectangle((y, rows - x - 1), 1, 1, color='cyan', label='Discrete Path' if (x, y) == path[0] else None))
        smooth = smooth_path(path)
        sx, sy = zip(*[(y + 0.5, rows - x - 1 + 0.5) for (x, y) in smooth])
        ax.plot(sx, sy, color='blue', linewidth=2, label='Smoothed Path')

    # Plot start (green) and goal (red)
    sx, sy = start
    gx, gy = goal
    ax.add_patch(plt.Rectangle((sy, rows - sx - 1), 1, 1, color='green', label='Start'))
    ax.add_patch(plt.Rectangle((gy, rows - gx - 1), 1, 1, color='red', label='Goal'))

    # Configure plot
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.set_aspect('equal')
    plt.title("Dijkstra's Pathfinding Visualization")
    plt.legend(loc='upper right')
    plt.show()

# Optional cost function (e.g., weighted terrain)
def terrain_cost(u, v):
    return 1

# Example usage
grid = [
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0],
    [0,1,0,1,0,1,0,1,1,0,1,0,1,1,0],
    [0,1,0,0,0,1,0,0,0,0,1,0,0,1,0],
    [0,0,0,1,0,0,0,1,1,0,0,0,0,0,0],
    [1,1,0,0,0,1,1,0,0,1,1,0,0,1,1],
    [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
    [0,1,1,1,1,1,0,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]

start = (0, 0)
goal = (7, 14)
path, total_cost = dijkstra(grid, start, goal, cost_fn=terrain_cost)
print("Total Cost:", total_cost)
plot_grid(grid, path, start, goal)
