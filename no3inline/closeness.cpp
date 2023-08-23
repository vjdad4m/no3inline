#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

/*
2 3
1 0 1
0 1 1
1 1 0

1 0 1
1 1 1
0 0 1

6
*/

struct pt{
    ll x;
    ll y;
};

ll devi(int n, vector<pt> &p1, vector<pt> &p2){
    vector<bool> vis1(n, 0), vis2(n, 0);
    int i1 = 0, i2 = 0;
    ll r = 0;
    while (i1 < n && i2 < n){
        while (i1 < n && vis1[i1]) i1++;
        while (i2 < n && vis2[i2]) i2++;
        if (i1 == n) break;
        int besti = 0, bestd = 1e9;
        if (p1[i1].x < p2[i1].x || p1[i1].x == p2[i1].x && p1[i1].y < p2[i1].y){
            //cout << 1 << " " << p1[i1].x << " " << p1[i1].y << "\n";
            for (int i=0; i<n; i++){
                if (vis2[i]) continue;
                int d = abs(p1[i1].x - p2[i].x) + abs(p1[i1].y - p2[i].y);
                if (d < bestd){
                    bestd = d;
                    besti = i;
                }
            }
            vis1[i1] = true;
            vis2[besti] = true;
            r += bestd;
        }
        else{
            //cout << 2 << " " << p2[i2].x << " " << p2[i2].y << "\n";
            for (int i=0; i<n; i++){
                if (vis1[i]) continue;
                int d = abs(p1[i].x - p2[i2].x) + abs(p1[i].y - p2[i2].y);
                if (d < bestd){
                    bestd = d;
                    besti = i;
                }
            }
            vis1[besti] = true;
            vis2[i2] = true;
            r += bestd;
        }
    }
    return r;
}

ll reward(int &n, int &sess, vector<vector<vector<bool> > > &m){
    ll r = 0;
    vector<vector<pt> > p(sess);
    for (int i=0; i<sess; i++){
        for (int j=0; j<n; j++){
            for (int l=0; l<n; l++){
                if (m[i][j][l]) p[i].push_back({j, l});
            }
        }
    }
    for (int i=0; i<sess; i++){
        for (int j=i+1; j<sess; j++){
            r += devi(2*n, p[i], p[j]);
        }
    }
    return r;
}

int main(){
    int sess, n;
    cin >> sess >> n;
    vector<vector<vector<bool> > > m(sess, vector<vector<bool> >(n, vector<bool>(n, 0)));
    bool b;
    for (int i=0; i<sess; i++){
        for (int j=0; j<n; j++){
            for (int l=0; l<n; l++){
                cin >> b;
                m[i][j][l] = b;
            }
        }
    }
    cout << reward(n, sess, m);
}