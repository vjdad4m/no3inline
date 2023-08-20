#include <stdio.h>
#include <stdlib.h>

typedef long long ll;

typedef struct {
    ll x;
    ll y;
} pt;

/**
 * Determine the orientation of the triplet (a, b, c) in 2D plane.
 *
 * @param a The first point.
 * @param b The second point.
 * @param c The third point.
 * @return 1 if the points make a counter-clockwise turn.
 *        -1 if the points make a clockwise turn.
 *         0 if the points are collinear.
 */
int turn(pt a, pt b, pt c) {
    if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) > 0) return 1;
    else if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) < 0) return -1;
    else return 0;
}

/**
 * Calculate the reward based on a matrix by finding the number of 
 * collinear points in the matrix.
 *
 * @param m The 2D matrix.
 * @param m_size The size of the matrix (assuming it's a square matrix).
 * @return The number of sets of 3 points that are collinear.
 */
int reward(int **m, int m_size) {
    // Initialize reward counter
    int r = 0;

    // Dynamic array to hold points from the matrix
    pt* p = NULL;
    int p_size = 0;

    // Traverse the matrix to populate the points array
    for (int i = 0; i < m_size; i++) {
        for (int j = 0; j < m_size; j++) {
            // If a value is true (non-zero) in the matrix, consider it as a point
            if (m[i][j]) {
                p = (pt*)realloc(p, (p_size + 1) * sizeof(pt));
                p[p_size].x = i;
                p[p_size].y = j;
                p_size++;
            }
        }
    }

    // Use a triple nested loop to check every combination of 3 points
    for (int i = 0; i < p_size; i++) {
        for (int j = i + 1; j < p_size; j++) {
            for (int l = j + 1; l < p_size; l++) {
                // If the three points are collinear, increase the reward
                if (!turn(p[i], p[j], p[l])) {
                    r++;
                }
            }
        }
    }

    // Clean up dynamic memory
    free(p);

    return r;
}