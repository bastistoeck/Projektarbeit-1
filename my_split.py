import numpy as np

def my_split(array, size): ### split array into lengths of size

    if array.shape[0] % size == 0: ### divisible

        return np.split(array, size)

    else: ## shave the end off then reattach:

        remainder = array.shape[0] % size

        shaved_array = array[:-remainder]

        split_array = np.split(shaved_array, size)

        split_array[-1] = np.concatenate([split_array[-1], array[-remainder:] ])

        return split_array