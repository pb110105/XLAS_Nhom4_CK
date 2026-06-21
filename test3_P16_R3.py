import numpy as np
from src.bai3 import circular_lbp_hand_calc

# Ma trận ảnh xám 9x9
img = np.array([
    [10,20,30,40,50,60,70,80,90],
    [15,25,35,45,55,65,75,85,95],
    [20,30,40,50,60,70,80,90,100],
    [25,35,45,55,65,75,85,95,105],
    [30,40,50,60,70,80,90,100,110],
    [35,45,55,65,75,85,95,105,115],
    [40,50,60,70,80,90,100,110,120],
    [45,55,65,75,85,95,105,115,125],
    [50,60,70,80,90,100,110,120,130]
], dtype=np.uint8)

lbp16 = circular_lbp_hand_calc(img, P=16, R=2)

rows, cols = lbp16.shape

lbp_final = np.zeros((rows, cols), dtype=np.uint8)

for y in range(rows):
    for x in range(cols):

        val = lbp16[y, x]

        part1 = val & 255
        part2 = (val >> 8) & 255

        lbp_final[y, x] = max(part1, part2)

print("\nLBP sau khi lấy max:")
print(lbp_final)

val = lbp_final[4,4]

part1 = val & 255
part2 = (val >> 8) & 255

print("LBP 16 bit =", bin(val))
print("Part 1 =", part1)
print("Part 2 =", part2)

print("Max =", max(part1, part2))