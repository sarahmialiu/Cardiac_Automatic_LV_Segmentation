from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.io import loadmat
import skimage.transform
import matplotlib.pyplot as plt

class PaddedDataset:
    def __init__(self, img_pathes: Path, mask_pathes: Path, intensity_min, intensity_max) -> None:
        self.img_pathes = img_pathes
        self.mask_pathes = mask_pathes
        self.slices = [nib.load(p).shape[-1] for p in self.img_pathes]
        self.cum_slices = np.cumsum(self.slices)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __getitem__(self, index: int):
        path_index = np.searchsorted(self.cum_slices, index, side='right')
        if path_index == 0:
            slice_index = index
        else:
            slice_index = index - self.cum_slices[path_index - 1]
        
        # Loading padded binary mask (1200, 1200)
        mat_file = loadmat(self.mask_pathes[path_index])
        mask = mat_file["binary"][:,:,slice_index]
        mask = mask[::2, ::2] # downsamples by quartering resolution
        # assert mask.shape == (400, 400), "Resized mask shape does not match desired shape"
        
        # Loading padded ultrasound image (1200, 1200)
        img = nib.load(self.img_pathes[path_index]).get_fdata()[:,:,slice_index]
        img = img[::2, ::2]

        # mask = mask.flatten() # flattening for LSTM
        # img = img.flatten()

        assert img.shape == mask.shape, "Resized image shape: {}, resized mask shape: {}".format(img.shape, mask.shape)
        
        img = windowing(img, self.intensity_min, self.intensity_max)[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        
        return img.astype(np.float32), mask.astype(np.float32)
     
    def filter_samples(self, index):
        path_index = np.searchsorted(self.cum_slices, index, side='right') # Check if the sum of the mask for the given index is greater than 0
        if path_index == 0:
            slice_index = index
        else:
            slice_index = index - self.cum_slices[path_index - 1]

        mat_file = loadmat(self.mask_pathes[path_index])
        mask = mat_file["binary"][:,:,slice_index]
        return np.sum(mask) > 0 # returns true if mask has values
    
    def __len__(self):
        return self.cum_slices[-1]
    
class UnpaddedDataset:
    def __init__(self, img_pathes: Path, mask_pathes: Path, intensity_min, intensity_max) -> None:
        self.img_pathes = img_pathes
        self.mask_pathes = mask_pathes
        self.slices = [nib.load(p).shape[-1] for p in self.img_pathes]
        self.cum_slices = np.cumsum(self.slices)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __getitem__(self, index: int):
        path_index = np.searchsorted(self.cum_slices, index, side='right')
        if path_index == 0:
            slice_index = index
        else:
            slice_index = index - self.cum_slices[path_index - 1]
        
        # Loading binary mask
        mat_file = loadmat(self.mask_pathes[path_index])
        mask = mat_file["binary"][:,:,slice_index]
        mask = skimage.transform.resize(mask, (600, 600))

        # Loading ultrasound image
        img = nib.load(self.img_pathes[path_index]).get_fdata()[:,:,slice_index]
        img = skimage.transform.resize(img, (600, 600))
        
        assert img.shape == mask.shape, "Resized image shape: {}, resized mask shape: {}".format(img.shape, mask.shape)
        
        #img = windowing(img, self.intensity_min, self.intensity_max)[np.newaxis, ...]
        img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        return img.astype(np.float32), mask.astype(np.float32)
    
    def __len__(self):
        return self.cum_slices[-1]

class VolumeDataset:
    def __init__(self, img_paths: Path, mask_paths: Path, intensity_min, intensity_max) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __getitem__(self, index: int):
        path_index = index

        img = nib.load(self.img_paths[path_index]).get_fdata()
        print(img.shape)
        img = np.transpose(img, (2,0,1)) # changes dimension ordering like so: (a, b, c) -> (c, a, b)
        print(img.shape)

        mask = loadmat(self.mask_paths[path_index])["binary"]
        mask = np.transpose(mask, (2,0,1))

        img = windowing(img, self.intensity_min, self.intensity_max)#[:, np.newaxis, ...]
        #mask = mask[:, np.newaxis, ...]

        return img.astype(np.float32), mask.astype(np.float32), path_index
    
    def __len__(self):
        return len(self.img_paths)



def windowing(image, min_value, max_value):
    image_new = np.clip(image, min_value, max_value)
    image_new = (image_new - min_value) / (max_value - min_value)
    return image_new

class Subset(PaddedDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]