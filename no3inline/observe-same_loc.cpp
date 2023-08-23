#include <bits/stdc++.h>
using namespace std;

long long sameLocSum(vector<vector<vector<bool>>> &grid){
    long long n=grid[0].length();
    vector<vector<long long>> matrix(n, vector<long long>(n));
    for(long long i=0; i<grid.length(); i++){
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

int main(){
    sameLocSum(grid);
}