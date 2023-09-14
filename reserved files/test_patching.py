import numpy as np

def create_sliding_patches(array, R):
    nx, ny, nz, nt = array.shape
    patch_size = (nx // R, ny // R, nz // R, nt)
    patches = []

    for x in range(0, nx - patch_size[0] + 1, patch_size[0]):
        for y in range(0, ny - patch_size[1] + 1, patch_size[1]):
            for z in range(0, nz - patch_size[2] + 1, patch_size[2]):
                for t in range(0, nt - patch_size[3] + 1, patch_size[3]):
                    patch = array[x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2], t : t + patch_size[3]]
                    patches.append(patch)

    return patches

# Example usage
nx, ny, nz, nt = 8, 8, 8, 100
R = 2
input_array = np.random.rand(nx, ny, nz, nt)

sliding_patches = create_sliding_patches(input_array, R)
print(len(sliding_patches))  # This will print the number of patches created
