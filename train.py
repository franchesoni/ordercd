import tqdm
import os
import time
import torch.nn.functional as F
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

HP, WP, P = 27, 27, 14

class Sentinel2Dataset(torch.utils.data.Dataset):
    def __init__(self, root="data/sentinel_2"):
        self.root = root
        self.samples = sorted(os.listdir(root))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root, self.samples[idx])
        sample = np.load(sample_path)[:, :, : P * HP, : P * WP]
        return sample


def pairwise_ranking_loss_vectorized(pred, margin=1.0):
    """
    pred: tensor of shape (B, T, T) with predicted deltas.
    For each anchor i, for all pairs (j, k) where |i-j| > |i-k|,
    penalizes if pred[i,j] isn't at least margin greater than pred[i,k].
    """
    B, T, _ = pred.shape
    device = pred.device
    # Create true differences: for each anchor i, true_diff[i, j] = |i - j|
    idx = torch.arange(T, device=device).unsqueeze(1)
    true_diff = torch.abs(idx - idx.t())  # (T, T)
    # For each anchor i, mask: valid if true_diff[i, j] > true_diff[i, k]
    # Expand dims: (T, T, T): for anchor i, compare true_diff[i, :].unsqueeze(2) vs. .unsqueeze(1)
    mask = (true_diff.unsqueeze(2) > true_diff.unsqueeze(1)).unsqueeze(
        0
    )  # (1, T, T, T)

    # Compute all differences: diff[b, i, j, k] = pred[b, i, j] - pred[b, i, k]
    diff = pred.unsqueeze(3) - pred.unsqueeze(2)  # (B, T, T, T)

    # Hinge loss only on valid pairs
    loss = F.relu(margin - diff)
    loss = (loss * mask).sum() / (mask.sum() * B)
    return loss


def main(n_epochs=1, seed=0, device="cpu"):
    torch.manual_seed(seed)
    writer = SummaryWriter()
    ds = Sentinel2Dataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    backbone.eval()
    backbone.to(device)

    transformer_layer = torch.nn.TransformerEncoderLayer(
        d_model=384, nhead=4, dim_feedforward=1024, dropout=0, batch_first=True
    )  # expects batch, seq, feat
    linear_layer = torch.nn.Linear(384, 1)
    transformer_layer.to(device)
    linear_layer.to(device)

    optim = torch.optim.Adam(
        list(transformer_layer.parameters()) + list(linear_layer.parameters()), lr=1e-4
    )
    times = {}
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(tqdm.tqdm(dl, total=len(dl))):
            batch = batch.to(device, non_blocking=True)
            batch = normalize(batch / 255.0)
            B, T, C, H, W = batch.shape
            st = time.time()
            with torch.no_grad():
                x = backbone.forward_features(batch.view(B * T, C, H, W))[
                    "x_norm_patchtokens"
                ].view(B, T, HP, WP, 384)
                x = x.unsqueeze(2) - x.unsqueeze(
                    1
                )  # B, T, T, HP, WP, 384, where dim 1 is i, dim 2 is j, and it contains feat[i] - feat[j]
            times["backbone"] = time.time() - st
            st = time.time()
            x = transformer_layer(
                x.view(B * T * T, HP * WP, 384)
            )  # (BTT, HPWP, 384),  these are "local change features"
            x = x.mean(
                dim=1
            )  # average pool over HPWP, (BTT, 384), these are "global change features"
            change_score = torch.sigmoid(
                linear_layer(x).view(B, T, T)
            )  # these are "distances"
            times["head"] = time.time() - st
            # optim step
            st = time.time()
            loss = pairwise_ranking_loss_vectorized(change_score, margin=0.01)
            times["loss"] = time.time() - st
            st = time.time()
            optim.zero_grad()
            loss.backward()
            optim.step()
            times["optim"] = time.time() - st
            # log
            writer.add_scalar("Loss/train", loss.item(), epoch * len(dl) + batch_idx)
            writer.add_scalars("times", times, epoch * len(dl) + batch_idx)
            print(loss.item(), times)
    writer.close()

    # save model
    torch.save(transformer_layer.state_dict(), os.path.join(writer.log_dir, "transformer_layer.pth"))
    torch.save(linear_layer.state_dict(), os.path.join(writer.log_dir, "linear_layer.pth"))



def visualize_sample(number):
    import numpy as np
    from PIL import Image

    sample_path = f"data/sentinel_2/{number}.npy"
    sample = np.load(sample_path)
    for number, frame in enumerate(sample):
        Image.fromarray(frame.transpose(1, 2, 0)).save(f"frame_{number}.jpeg")


if __name__ == "__main__":
    main()
