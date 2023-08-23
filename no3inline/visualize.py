import matplotlib.pyplot as plt
import numpy as np
from math import gcd

def visualize_grid(grid):
    grid = np.array(grid)
    n, m = grid.shape

    lines = read_lines()

    fig, ax = plt.subplots(figsize=(8,8))
    for i in range(n):
        for j in range(m):
            if grid[i, j] == 1:
                ax.plot(j, i,"bo")

    for line in lines:
        a, b, c = line.a, line.b, line.c
        if b != 0:
            x1, x2 = -0.5, m + 0.5
            y1 = (-a * x1 - c) / b
            y2 = (-a * x2 - c) / b
        else:
            x1 = x2 = -c / a
            y1, y2 = -0.5, n + 0.5
        ax.plot([x1, x2], [y1, y2], 'r-')

    ax.set_xlim(-0.5, m-0.5)
    ax.set_ylim(n-0.5, -0.5)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_aspect('equal')

    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)

    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.set_yticklabels(range(n))

    plt.show()

def read_lines(filename="lines.txt"):
    lines = []
    with open(filename, "r") as file:
        for line in file:
            a, b, c = map(int, line.strip().split())
            lines.append(Line(a, b, c))
    return lines

class Line:
    def __init__(self, a, b, c):
        g = gcd(a, gcd(b, c))
        self.a = a // g
        self.b = b // g
        self.c = c // g
        if self.a < 0 or (self.a == 0 and self.b < 0):
            self.a = -self.a
            self.b = -self.b
            self.c = -self.c

grid = [
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]
]

visualize_grid(grid)
