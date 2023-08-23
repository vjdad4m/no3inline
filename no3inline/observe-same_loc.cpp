#include "observe-same_loc.h"

#define ll long long

long long sameLocSum(const vector<bool> &flattened_grid, int dim1, int dim2, int dim3){
    vector<vector<vector<bool>>> grid(dim1, vector<vector<bool>>(dim2, vector<bool>(dim3)));
    int idx = 0;

    for(int i = 0; i < dim1; i++){
        for(int j = 0; j < dim2; j++){
            for(int k = 0; k < dim3; k++){
                grid[i][j][k] = flattened_grid[idx++];
            }
        }
    }

    long long n=grid[0].size();
    vector<vector<long long>> matrix(n, vector<long long>(n));
    for(long long i=0; i<grid.size(); i++){
        for(long long j=0; j<n; j++){
            for(long long k=0; k<n; k++){
                matrix[j][k]+=grid[i][j][k];
            }
        }
    }
    ll ans=0;
    for(long long i=0; i<n; i++){
        for(long long j=0; j<n; j++){
            ans+=matrix[i][j]*(matrix[i][j]-1)/2;
        }
    }
}
