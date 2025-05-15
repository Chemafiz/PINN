import numpy as np
from numba import njit, prange
import time
import matplotlib.pyplot as plt


def downsample(data, target_len):
    original_len = len(data)
    scale = original_len / target_len
    downsampled = np.zeros(target_len, dtype=np.float32)

    for i in range(target_len):
        start = int(np.floor(i * scale))
        end = int(np.floor((i + 1) * scale))

        # Liczba punktów w przedziale
        count = end - start if end > start else original_len - start + end

        # Zbieramy wartości z zawinięciem
        indices = [(start + j) % original_len for j in range(count)]
        downsampled[i] = np.mean([data[idx] for idx in indices])

    return downsampled


def upsample(data, target_len):
    original_len = len(data)
    scale = original_len / target_len
    upsampled = np.zeros(target_len, dtype=np.float32)

    for i in range(target_len):
        pos = i * scale
        idx_low = int(np.floor(pos)) % original_len
        idx_high = (idx_low + 1) % original_len

        weight_high = pos - np.floor(pos)
        weight_low = 1.0 - weight_high

        upsampled[i] = data[idx_low] * weight_low + data[idx_high] * weight_high

    return upsampled

@njit
def dupa(arr):
    return np.count_nonzero(np.isnan(arr), axis=1)


if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    y = np.sin(x)

    y_down = downsample(y, 90)
    x_down = np.linspace(0, 2 * np.pi, 90, endpoint=False)

    x = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    y = np.sin(x)

    y_up= upsample(y, 100)
    x_up = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    

    # plt.plot(x, y)
    # # plt.scatter(x_down, y_down, s=1)
    # plt.scatter(x_up, y_up, s=1)
    # plt.show()

    temp = np.array([
        [1,np.nan,3],
        [1,2,3],
        [1,np.nan,3],
        ])
    print(dupa(temp))