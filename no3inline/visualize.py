import matplotlib.pyplot as plt
import numpy as np

def visualize_grid(grid, filename=None, lines=[]):
    grid = np.array(grid)
    n, m = grid.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n):
        for j in range(m):
            if grid[i, j] == 1:
                ax.plot(j, i, "bo")

    for line in lines:
        a, b, c = line
        if b != 0:
            x1, x2 = -0.5, m + 0.5
            y1 = (-a * x1 - c) / b
            y2 = (-a * x2 - c) / b
        else:
            x1 = x2 = -c / a
            y1, y2 = -0.5, n + 0.5
        ax.plot([x1, x2], [y1, y2], 'r-')

    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_aspect('equal')

    # Set grid and axis linewidth
    linewidth = 0.5
    ax.grid(which='both', color='gray', linestyle='-', linewidth=linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)

    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Remove the numbers from the axis
    ax.set_xticklabels(['' for _ in range(m)])
    ax.set_yticklabels(['' for _ in range(n)])

    if filename is not None:
        plt.savefig(filename, dpi=200)
        return fig
    else:
        plt.show()

def main():
    grid = [
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]
    ]

    lines = [
        [0, 1, -1],
        [1, -1, 1],
        [1, 0, -2],
        [1, 1, -3]
    ]
    visualize_grid(grid, None, lines)

if __name__ == '__main__':
    main()
