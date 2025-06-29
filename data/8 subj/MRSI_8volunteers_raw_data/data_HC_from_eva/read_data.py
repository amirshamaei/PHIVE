import os
import h5py
import numpy as np
import mat73 as mat
import tqdm
def read(rng):
    csi = []
    mask = []
    # Iterate through folders HC01_M01 to HC08_M01
    for folder_num in tqdm.tqdm(rng):
        folder_name = f"HC0{folder_num}_M01"
        folder_path = os.path.join(root_folder, folder_name)

        # Initialize a list to store data from each folder


        # Iterate through files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mat'):
                file_path = os.path.join(folder_path, file_name)

                # Load MATLAB v7.3 file using h5py.File
                # Load MATLAB v7.3 file using h5py.File
                mat_file = mat.loadmat(file_path)
                # Assuming the data is stored under a specific dataset name 'data'
                csi_ = mat_file['csi'][:]
                csi.append(csi_)
                mask_ = mat_file['mask'][:]
                mask.append(mask_)
    return csi,mask

def read_mat_files_from_folders(root_folder,rng = range(1, 8)):
    csi, mask =read(rng)
    gathered = []
    sizes = []
    for csi_ , mask_ in zip(csi,mask):
        gathered.append(csi_[mask_.astype(bool),:])
        sizes.append(np.count_nonzero(mask_))
    return gathered,sizes

def read_mat_files_from_folders_test(root_folder, rng=range(1, 8)):
    csi ,mask = read(rng)
    gathered = []
    for csi_ , mask_ in zip(csi,mask):
        gathered.append(csi_*np.expand_dims(mask_,-1))
    return gathered,mask

if __name__ == "__main__":
    root_folder = ""  # Replace with the path to your main folder containing HC0x_M01 subfolders
    all_data,sizes = read_mat_files_from_folders(root_folder,rng=range(1, 8))
    # np.save('sizes', np.asarray(sizes), allow_pickle=True)
    # np.save('train_data',np.concatenate(all_data,0),allow_pickle=True)

    all_data ,all_masks = read_mat_files_from_folders_test(root_folder,rng=range(8,10))
    for i, (data, mask) in enumerate(zip(all_data,all_masks)):
        np.save(f'test_data_{i}', (data), allow_pickle=True)
        np.save(f'test_mask_{i}', (mask), allow_pickle=True)
    # print(all_data)
