import numpy as np


pattern_text = [
    "...o.o.",
    "...oo..",
    "o.....o",
    ".o..o..",
    ".....oo",
    ".oo....",
    "o.o...."
]

pattern_array = np.array([[1 if char == "o" else 0 for char in row] for row in pattern_text])

print(pattern_array)
np.savetxt("counterex_7x7.txt", pattern_array, delimiter=', ', fmt='%s')
