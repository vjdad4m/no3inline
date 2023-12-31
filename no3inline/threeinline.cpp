#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <fstream>

using namespace std;

struct pont {
    long long x, y;
    pont operator - (const pont& a) const {
        return pont{x - a.x, y - a.y};
    }
};

struct Line {
    long long a, b, c;

    // Normalize the coefficients by dividing them with their GCD.
    void normalize() {
        long long g = __gcd(a, __gcd(b, c));
        a /= g;
        b /= g;
        c /= g;
        if (a < 0 || (a == 0 && b < 0)) {
            a = -a;
            b = -b;
            c = -c;
        }
    }

    bool operator<(const Line &o) const {
        if (a != o.a) return a < o.a;
        if (b != o.b) return b < o.b;
        return c < o.c;
    }
};

Line getLine(const pont& p1, const pont& p2) {
    Line l;
    l.a = p2.y - p1.y;
    l.b = p1.x - p2.x;
    l.c = -(l.a * p1.x + l.b * p1.y);
    l.normalize();
    return l;
}
void writeLines(const set<Line> &lines) {
    ofstream out("lines.txt");
    for (const Line &line : lines) {
        out << line.a << " " << line.b << " " << line.c << "\n";
    }
    out.close();
}

int solve(int &n, vector<pont> &p) {
    
    set<Line> lines;
    set<Line> dupLines;
    for (int i = 0; i < 2 * n; i++) {
        for (int j = i + 1; j < 2 * n; j++) {
            Line theLine=getLine(p[i], p[j]);
            if(lines.find(theLine)!=lines.end()){
                dupLines.insert(theLine);
            }
            else{
                lines.insert(theLine);
            }
        }
    }
    writeLines(dupLines);
    return (2*n)*(2*n-1)/2-lines.size();
}

int main() {
    cout << "Distinct lines: " << solve() << "\n";
    return 0;
}
