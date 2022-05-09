import numpy as np


# ----------q1----------
def change_sign():
    array = np.array(range(21))
    np.copysign(array[9:16], -1)
    return array


if __name__ == "__main__":
    print(change_sign)
