import numpy as np
from src.Bai2 import create_I1_I2_I3
from src.Bai2 import median_filter_manual

img = np.array([
    [10,20,30,40,50],
    [20,30,40,50,60],
    [30,40,50,60,70],
    [40,50,60,70,80],
    [50,60,70,80,90]
], dtype=np.uint8)

I1, I2, I3 = create_I1_I2_I3(img)

print("I1")
print(I1)

print("I2")
print(I2)

print("I3")
print(I3)

# I4 = Median(I3, 3x3)
I4 = median_filter_manual(I3, kernel_size=3)

# I5 = Median(I1, 5x5)
I5 = median_filter_manual(I1, kernel_size=5)

print("I4")
print(I4)

print("I5")
print(I5)

# Padding để cùng kích thước
h = max(I4.shape[0], I5.shape[0])
w = max(I4.shape[1], I5.shape[1])

I4_pad = np.zeros((h, w), dtype=np.uint8)
I5_pad = np.zeros((h, w), dtype=np.uint8)

I4_pad[:I4.shape[0], :I4.shape[1]] = I4
I5_pad[:I5.shape[0], :I5.shape[1]] = I5

# I6
I6 = np.where(I4_pad > I5_pad, 0, I5_pad)

print("I6")
print(I6)