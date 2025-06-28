import os
import h5py
import numpy as np
import mat73 as mat
import tqdm
def read(rng):
    csi = []
    mask = []
    folder_names= []
    # Iterate through folders HC01_M01 to HC08_M01
    for folder_num in tqdm.tqdm(rng):
        folder_name = f"MS_Patient_{245+folder_num}"
        folder_path = os.path.join(root_folder, folder_name)
        folder_names.append(folder_name)
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

    return csi,mask,folder_names

def read_mat_files_from_folders(root_folder,rng = range(1, 8)):
    csi, mask,folder_names =read(rng)
    gathered = []
    sizes = []
    for csi_ , mask_ in zip(csi,mask):
        gathered.append(csi_[mask_.astype(bool),:])
        sizes.append(np.count_nonzero(mask_))
    return gathered,sizes,folder_names

def read_mat_files_from_folders_test(root_folder, rng=range(1, 8)):
    csi ,mask,folder_names = read(rng)
    gathered = []
    for csi_ , mask_ in zip(csi,mask):
        gathered.append(csi_*np.expand_dims(mask_,-1))
    return gathered,mask,folder_names

if __name__ == "__main__":
    root_folder = ""  # Replace with the path to your main folder containing HC0x_M01 subfolders
    # all_data,sizes = read_mat_files_from_folders(root_folder,rng=range(1, 8))
    # np.save('sizes', np.asarray(sizes), allow_pickle=True)
    # np.save('train_data',np.concatenate(all_data,0),allow_pickle=True)

    all_data ,all_masks,folder_names = read_mat_files_from_folders_test(root_folder,rng=range(0,3))
    for i, (data, mask) in zip(folder_names,zip(all_data,all_masks)):
        np.save(f'{i}', (data), allow_pickle=True)
        np.save(f'{i}_mask', (mask), allow_pickle=True)
    # print(all_data)
