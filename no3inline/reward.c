#include <stdio.h>
#include <stdlib.h>

typedef long long ll;

typedef struct {
    ll x;
    ll y;
} pt;

int turn(pt a, pt b, pt c) {
    if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) > 0) return 1;
    else if ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y) < 0) return -1;
    else return 0;
}

int reward(int **m, int m_size) {
    int r = 0;
    pt* p = NULL;
    int p_size = 0;

    for (int i = 0; i < m_size; i++) {
        for (int j = 0; j < m_size; j++) {
            if (m[i][j]) {
                p = (pt*)realloc(p, (p_size + 1) * sizeof(pt));
                p[p_size].x = i;
                p[p_size].y = j;
                p_size++;
            }
        }
    }
    for (int i = 0; i < p_size; i++) {
        for (int j = i + 1; j < p_size; j++) {
            for (int l = j + 1; l < p_size; l++) {
                if (!turn(p[i], p[j], p[l])) {
                    r++;
                }
            }
        }
    }
    free(p);
    return r;
}
