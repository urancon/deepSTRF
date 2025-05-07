import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(ROOT_DIR)

from datasets.CRCNS_PVC1_Dataset import CRCNS_PVC1_Dataset
from models.video.video_zoo import VideoModel1
from metrics.performance import normalized_correlation_coefficient, correlation_coefficient, signal_power

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) backend:", device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU:", device)
else:
    device = torch.device("cpu")
    print("Using CPU:", device)

## Hyperparameters ##

BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_LR = 2 * LR
EPOCHS = 100

spatial_resol=(112, 112)

SIGNAL_POWER_THRESHOLD = 0.1

## Load the dataset ##

dataset_train = CRCNS_PVC1_Dataset(
    path='datasets/CRCNS_PVC1',
    set='train',
    spatial_resol=spatial_resol,
    grayscale=True,
    normalize_videos=False,
    normalize_responses=False,
    response_smoothing=False,
    add_noise=False,
    temporal_resolution=10,
    split_into_clips=False,
    new_clip_len=300
)

dataset_test = CRCNS_PVC1_Dataset(
    path='datasets/CRCNS_PVC1',
    set='test',
    spatial_resol=spatial_resol,
    grayscale=True,
    normalize_videos=False,
    normalize_responses=False,
    response_smoothing=False,
    add_noise=False,
    temporal_resolution=10,
    split_into_clips=False,
    new_clip_len=300
)

print("Dataset loaded successfully.")
print("Videos shape:", dataset_train.videos.shape)      # Expected shape (S, C, H, W, T)
print("Responses shape:", dataset_train.responses.shape)  # Expected shape (N, S, R, T)

train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

## Select neurons ##
neuron_indices = list(range(dataset_train.N_neurons))
dataset_train.select_population(neuron_indices)
dataset_test.select_population(neuron_indices)
print("Selected neurons:", neuron_indices)

## Split dataset ##
total_train = len(dataset_train)
n_train = int(0.7 * total_train)
n_val = int(total_train - n_train)

train_set, val_set = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(dataset_test)}")

## Define model ##

out_neurons = dataset_train.responses.shape[0]  # total number of neurons
print(f'We are fitting {out_neurons} neurons')
model = VideoModel1(spatial_resolution=spatial_resol, out_neurons=out_neurons)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total model parameters: {total_params:,}')

## Loss and optimizer ##
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    anneal_strategy='linear'  # or 'cos' depending on your preference
)

## Training loop ##

if __name__ == "__main__":
    print("Training started...")
    print(f"Using device: {device}")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_cc_total = 0.0
        train_ccnorm_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for video, response, _, _ in pbar:
            video = video.to(device)                    # (B, C, H, W, T)
            response = response.to(device)              # (B, N, R, T)
            
            # Mask to get where signal power is above the threshold
            sp = signal_power(response)
            mask = (sp > SIGNAL_POWER_THRESHOLD).float()  # (B, N)
            mask = mask.to(device)
            
            target = response.mean(dim=2)  # (B, N, T)
            target = target * mask.unsqueeze(-1) 

            optimizer.zero_grad()
            output = model(video)                       # (B, N, T)
            
            # Loss computation 
            loss_per_neuron = F.mse_loss(output, target, reduction='none')  # (B, N, T)
            loss_per_neuron = loss_per_neuron.mean(dim=-1)  # (B, N)
            masked_loss = loss_per_neuron * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-8)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Compute raw CC and normalized CC (ccnorm)
            cc = correlation_coefficient(output, target, reduction="none")  # (B, N)
            masked_response = response * mask.unsqueeze(-1).unsqueeze(-1)
            ccnorm = normalized_correlation_coefficient(
                y_pred=output,
                y_gt=masked_response,
                method='schoppe',
                reduction='none'  # (B, N)
            )

            masked_cc = cc * mask
            masked_ccnorm = ccnorm * mask

            total_valid = mask.sum() + 1e-8  

            avg_cc = masked_cc.sum() / total_valid
            avg_ccnorm = masked_ccnorm.sum() / total_valid

            train_loss += loss.item()
            train_cc_total += avg_cc.item()
            train_ccnorm_total += avg_ccnorm.item()
            pbar.set_postfix(loss=loss.item(), cc=avg_cc.item(), ccnorm=avg_ccnorm.item())

            model.reset_model()

        train_loss /= len(train_loader)
        train_cc_avg = train_cc_total / len(train_loader)
        train_ccnorm_avg = train_ccnorm_total / len(train_loader)

        model.eval()
        val_loss, val_cc_total, val_ccnorm_total = 0.0, 0.0, 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
            for video, response, _, _ in pbar:
                video = video.to(device)
                response = response.to(device)          # (B, N, R, T)
                
                # Mask to get where signal power is above the threshold
                sp = signal_power(response)
                mask = (sp > SIGNAL_POWER_THRESHOLD).float()  # (B, N)
                mask = mask.to(device)
                
                target = response.mean(dim=2)  # (B, N, T)
                target = target * mask.unsqueeze(-1) 
                
                output = model(video)                       # (B, N, T)
                
                # Loss computation 
                loss_per_neuron = F.mse_loss(output, target, reduction='none')  # (B, N, T)
                loss_per_neuron = loss_per_neuron.mean(dim=-1)  # (B, N)
                masked_loss = loss_per_neuron * mask
                loss = masked_loss.sum() / (mask.sum() + 1e-8)

                # Compute raw CC and normalized CC (ccnorm)
                cc = correlation_coefficient(output, target, reduction="none")  # (B, N)
                masked_response = response * mask.unsqueeze(-1).unsqueeze(-1)
                ccnorm = normalized_correlation_coefficient(
                    y_pred=output,
                    y_gt=masked_response,
                    method='schoppe',
                    reduction='none'  # (B, N)
                )

                masked_cc = cc * mask
                masked_ccnorm = ccnorm * mask

                total_valid = mask.sum() + 1e-8  

                avg_cc = masked_cc.sum() / total_valid
                avg_ccnorm = masked_ccnorm.sum() / total_valid

                val_loss += loss.item()
                val_cc_total += avg_cc.item()
                val_ccnorm_total += avg_ccnorm.item()
                pbar.set_postfix(loss=loss.item(), cc=avg_cc.item(), ccnorm=avg_ccnorm.item())

                model.reset_model()

        val_loss /= len(val_loader)
        val_cc_avg = val_cc_total / len(val_loader)
        val_ccnorm_avg = val_ccnorm_total / len(val_loader)

        print(f"[Epoch {epoch+1:02d}] "
              f"Train Loss: {train_loss:.4f} | Train CC: {train_cc_avg:.3f} | Train CCnorm: {train_ccnorm_avg:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val CC: {val_cc_avg:.3f} | Val CCnorm: {val_ccnorm_avg:.3f}")
        
    # ======== Final Test Evaluation ========
    
    model.eval()
    test_loss, test_cc_total, test_ccnorm_total = 0.0, 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Test", leave=False)
        for video, response, _, _ in pbar:
            video = video.to(device)
            response = response.to(device)          # (B, N, R, T)
            
            # Mask to get where signal power is above the threshold
            sp = signal_power(response)
            mask = (sp > SIGNAL_POWER_THRESHOLD).float()  # (B, N)
            mask = mask.to(device)
            
            target = response.mean(dim=2)  # (B, N, T)
            target = target * mask.unsqueeze(-1) 
            
            output = model(video)                       # (B, N, T)
            
            # Loss computation 
            loss_per_neuron = F.mse_loss(output, target, reduction='none')  # (B, N, T)
            loss_per_neuron = loss_per_neuron.mean(dim=-1)  # (B, N)
            masked_loss = loss_per_neuron * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-8)

            # Compute raw CC and normalized CC (ccnorm)
            cc = correlation_coefficient(output, target, reduction="none")  # (B, N)
            masked_response = response * mask.unsqueeze(-1).unsqueeze(-1)
            ccnorm = normalized_correlation_coefficient(
                y_pred=output,
                y_gt=masked_response,
                method='schoppe',
                reduction='none'  # (B, N)
            )

            masked_cc = cc * mask
            masked_ccnorm = ccnorm * mask

            total_valid = mask.sum() + 1e-8  

            avg_cc = masked_cc.sum() / total_valid
            avg_ccnorm = masked_ccnorm.sum() / total_valid

            test_loss += loss.item()
            test_cc_total += avg_cc.item()
            test_ccnorm_total += avg_ccnorm.item()
            pbar.set_postfix(loss=loss.item(), cc=avg_cc.item(), ccnorm=avg_ccnorm.item())

            model.reset_model()

    test_loss /= len(test_loader)
    test_cc_avg = test_cc_total / len(test_loader)
    test_ccnorm_avg = test_ccnorm_total / len(test_loader)

    print(f"\n[TEST RESULTS] "
        f"Loss: {test_loss:.4f} | "
        f"CC: {test_cc_avg:.3f} | "
        f"CCnorm: {test_ccnorm_avg:.3f}")
