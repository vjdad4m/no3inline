import matplotlib.pyplot as plt
import numpy as np

def visualize_grid(grid, filename=None):
    grid = np.array(grid)
    n, m = grid.shape

    fig, ax = plt.subplots(figsize=(8,8))
    for i in range(n):
        for j in range(m):
            if grid[i, j] == 1:
                ax.plot(j, i,"bo")
    
    ax.set_xlim(-0.5, m-0.5 )
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
    if filename is not None:
        plt.savefig(filename, dpi=200)
    else:
        plt.show()

def main():
    grid = [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ]

    visualize_grid(grid)

if __name__ == "__main__":
    main()