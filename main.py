from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import torch
from torch import optim, Tensor
from torch.nn import (
    functional as F,
    Module,
    utils
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

import dataset
import models
import dataloader

class TqdmExtraFormat(tqdm):
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time))
        return d
    
def plot(img, mask_true, mask_pred, idx):
    img = np.squeeze(img, axis=1)
    mask_true = np.squeeze(mask_true, axis=1)
    mask_pred = np.squeeze(mask_pred, axis=1)
    max_value = img.max()
    img /= max_value

    num_slice = img.shape[0]
    fig, axs = plt.subplots(num_slice, 3)
    axs: list[list[plt.Axes]]
    for i in range(num_slice):
        axs[i][0].imshow(img[i], cmap='gray')
        axs[i][0].set_title(f'z = {i}')
        axs[i][1].imshow(mask_true[i], cmap='gray')
        axs[i][1].set_title(f'ID {idx:03d}\nGround truth')
        axs[i][2].imshow(mask_pred[i], cmap='gray')
        axs[i][2].set_title('AI generated')\
        
        # plt.imshow(mask_true[i])
        # plt.show()
    for i in range(num_slice):
        for j in range(3):
            axs[i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return fig

def train(img_path, mask_path, debug):

    print('Training...')
    mask_train_val_pathes = sorted(Path(mask_path).glob('*unpadded_wall_mask.mat'))
    img_train_val_pathes = sorted(Path(img_path).glob('*half_sequence.nii'))
    
    print(f'len(img_train_val_pathes) = {len(img_train_val_pathes)}')
    print(f'len(mask_train_val_pathes) = {len(mask_train_val_pathes)}')

    train_size = 0.9
    random_seed = 230620

    img_train_pathes, img_val_pathes, mask_train_pathes, mask_val_pathes = train_test_split(
        img_train_val_pathes, mask_train_val_pathes, train_size=train_size, random_state=random_seed
    )

    print(f'len(img_train_pathes) = {len(img_train_pathes)}')
    print(f'len(img_val_pathes) = {len(img_val_pathes)}')
    print(f'len(mask_train_pathes) = {len(mask_train_pathes)}')
    print(f'len(mask_val_pathes) = {len(mask_val_pathes)}')

    intensity_min = 30
    intensity_max = 100
    trainset = dataset.UnpaddedDataset(img_train_pathes, mask_train_pathes, intensity_min, intensity_max)
    valset = dataset.UnpaddedDataset(img_val_pathes, mask_val_pathes, intensity_min, intensity_max)
    print(f'len(trainset) = {len(trainset)}')
    print(f'len(valset) = {len(valset)}')

    # train_batch_sampler = dataloader.CustomBatchSampler(img_train_pathes)
    # val_batch_sampler = dataloader.CustomBatchSampler(img_val_pathes)

    # trainloader = DataLoader(trainset, batch_sampler=train_batch_sampler, num_workers=1) # first batch is size: (21, 1, 600, 600)
    # valloader = DataLoader(valset, batch_sampler=val_batch_sampler, num_workers=1)


    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)
    valloader = DataLoader(valset, batch_size=4, shuffle=True, num_workers=1)

    print(f'len(trainloader) = {len(trainloader)}')
    print(f'len(valloader) = {len(valloader)}')

    device = "cuda:0"
    model_class = getattr(models, "U_Net")
    model = model_class() # for LSTM, pass in the device
    
    print(f'model.__class__.__name__ = {model.__class__.__name__}')
    print()

    model.to(device)

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=1 / 10 ** .5, patience=2, verbose=True)
    
    val_metrics = Path("out/val_metrics.txt")
    if val_metrics.is_file():
        val_metrics.unlink() 
    
    torch.cuda.empty_cache()
    gc.collect()

    for epoch in range(1, 6): 
        if Path("out/temp_model_weights.pth").is_file(): 
            model.load_state_dict(torch.load("out/temp_model_weights.pth"))
        print(f'epoch = {epoch:03d}')

        train_epoch(trainloader, model, optimizer, debug)
        torch.cuda.empty_cache()
        gc.collect()

        val_epoch(valloader, model, scheduler, debug)
        torch.cuda.empty_cache()
        gc.collect()
        print()

        if debug and epoch == 2:
            break
        if optimizer.param_groups[0]['lr'] < 1e-6:
            break

def train_epoch(loader: DataLoader, model: Module, optimizer: optim.Adam, debug):
    print(f'length of train loader = {len(loader)}')
    model.train()
    device = "cuda:0"

    loss_values = list()
    
    with tqdm(total=len(loader)) as pbar:
        for batch_idx, sample in enumerate(loader): 
            sample: tuple[Tensor, ...]
            img, mask_true = sample # img, mask_true, path_idx = sample
            
            model_output = model(img.to(device)) 
             
            mask_pred = model_output.float().to(device).requires_grad_()
            mask_true = (mask_true > 0).float().to(device).requires_grad_()

            # Loss function defined here
            intersection = torch.sum(mask_pred * mask_true)
            union = torch.sum(mask_pred) + torch.sum(mask_true)
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            loss = 1. - dice
            
            optimizer.zero_grad()
            loss.backward()
            loss_values.append(loss.item())
            optimizer.step()
            
            pbar.update()
            if debug and batch_idx == 1:
                    break
                    
    torch.save(model.state_dict(), Path("out/temp_model_weights.pth"))
    
    pbar.close()
    print(f'loss = {np.mean(loss_values)} ± {np.std(loss_values)}')

def val_epoch(loader: DataLoader, model: Module, scheduler: optim.lr_scheduler.ReduceLROnPlateau, debug):
    print(f'length of validation loader = {len(loader)}')
    model.eval()
    device = "cuda:0"
    weights_save_path = Path("out/model_weights.pth")
    val_metrics_path = Path("out/val_metrics.txt")
    new_metrics = []
    
    if not val_metrics_path.is_file():
        old_metrics = [0]
    else:
        old_metrics = np.loadtxt(val_metrics_path)
    
    with tqdm(total=len(loader)) as pbar:
        for idx, sample in enumerate(loader):
            sample: tuple[Tensor, ...]
            img, mask_true = sample
            model_output = model(img.to(device))
            
            mask_pred = model_output.cpu().detach().numpy()
            mask_true = (mask_true > 0).numpy()
            dice = 1 - distance.dice(mask_pred.reshape(-1), mask_true.reshape(-1))
            new_metrics.append(dice)
            pbar.update()

            if debug and idx == 1:
                break

    pbar.close()
    
    print(f'dice = {np.mean(new_metrics)} ± {np.std(new_metrics)}')
    
    if np.mean(new_metrics) > np.mean(old_metrics) or not weights_save_path.is_file():
        print("Performance improved, saving new weights.")
        torch.save(model.state_dict(), Path("out/model_weights.pth"))
        
    scheduler.step(np.mean(new_metrics))
    np.savetxt(val_metrics_path, new_metrics, fmt='%.5f')

def test(img_path, mask_path, debug):
    print('Testing...')
    
    mask_test_pathes = sorted(Path(mask_path).glob('*unpadded_wall_mask.mat'))
    img_test_pathes = sorted(Path(img_path).glob('*half_sequence.nii'))
    weights_path = "out/model_weights.pth"

    print(f'len(img_test_pathes) = {len(img_test_pathes)}')
    print(f'len(mask_test_pathes) = {len(mask_test_pathes)}')

    intensity_min = 30
    intensity_max = 100

    testset = dataset.UnpaddedDataset(img_test_pathes, mask_test_pathes, intensity_min, intensity_max)
    print(f'len(testset) = {len(testset)}')

    test_batch_sampler = dataloader.CustomBatchSampler(img_test_pathes)

    testloader = DataLoader(testset, batch_sampler=test_batch_sampler, num_workers=1) # first batch is size: (21, 1, 600, 600)
    print(f'len(testloader) = {len(testloader)}')
        
    model_class = getattr(models, "U_Net")
    model = model_class()
    
    print(f'model.__class__.__name__ = {model.__class__.__name__}')
    print()
    
    model.load_state_dict(torch.load(weights_path))
    device = "cuda:0"
    model.to(device)

    test_metrics_path = Path("out/test_metrics.txt")
    new_metrics = []
    pdf = PdfPages('out/test_figures.pdf')

    with torch.no_grad(), tqdm(total=len(testloader)) as pbar:
        for idx, sample in enumerate(testloader):
            sample: tuple[Tensor, ...]
            img, mask_true = sample
            
            model_output = model(img.to(device))
            mask_pred = model_output.detach().cpu().numpy()
            mask_true = (mask_true > 0).numpy()
            
            dice = 1 - distance.dice(mask_pred.reshape(-1), mask_true.reshape(-1))
            new_metrics.append(dice)
            pbar.update()
            
            fig = plot(img.numpy(), mask_true, mask_pred, idx)
            fig.set_size_inches(15, 5 * mask_true.shape[0])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            if debug and idx == 1:
                break
    pbar.close()
    pdf.close()
    print(new_metrics)
    np.savetxt(test_metrics_path, new_metrics, fmt='%.5f')



img_train_2CH = "CAMUS\img_train_val_2CH"
mask_train_2CH = "CAMUS\mask_train_val_2CH"

img_train_4CH = "CAMUS\img_train_val_4CH"
mask_train_4CH = "CAMUS\mask_train_val_4CH"

img_test_2CH = "CAMUS\img_test_2CH"
mask_test_2CH = "CAMUS\mask_test_2CH"

img_test_4CH = "CAMUS\img_test_4CH"
mask_test_4CH = "CAMUS\mask_test_4CH"

if __name__ == '__main__':
    train(img_train_2CH, mask_train_2CH, debug=False)
    test(img_test_2CH, mask_test_2CH, debug=False)