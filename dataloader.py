from torch.utils.data import Sampler
from pathlib import Path
import nibabel as nib

def calc_batches(paths):
    # Create a list to hold the batches
    batches = []
    data_indices = list(range(10000))
    # random.shuffle(data_indices)

    for path_idx in range(len(paths)):
        # Randomly choose batch size within the specified range
        scan = nib.load(paths[path_idx]).get_fdata()
        batch_size = scan.shape[-1]
        batch_indices = data_indices[:batch_size]
        batches.append(batch_indices)
        data_indices = data_indices[batch_size:]
    
    # for batch in batches: print(len(batch))

    return batches

# def saved_batches(folder_path):
#     pass

class CustomBatchSampler(Sampler):
    def __init__(self, paths):
        self.paths = paths
        self.batches = calc_batches(paths)
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)
    
