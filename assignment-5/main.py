import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
LATENT_DIM_1 = 128  # Dimension of the first latent layer (z1)
LATENT_DIM_2 = 64  # Dimension of the second latent layer (z2)
IMAGE_SIZE = 256
BETA = 1.0  # Weight for the KL divergence term
IMAGE_PATH = "./datasets/Images"
OUTPUT_PATH = "output/hvae"
EPOCHS = 50

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "reconstructions"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "samples"), exist_ok=True)


def kl_divergence_gaussian(q_mean, q_logvar, p_mean=None, p_logvar=None):
    """
    Computes KL divergence KL(q || p) between two diagonal Gaussian distributions.
    If p is not specified, assumes p is N(0, I).
    """
    if p_mean is None or p_logvar is None:
        # KL divergence KL(q || N(0, I))
        kl_div = -0.5 * torch.sum(1 + q_logvar - q_mean.pow(2) - q_logvar.exp(), dim=1)
    else:
        # KL divergence KL(q || p)
        var_q = q_logvar.exp()
        var_p = p_logvar.exp()
        kl_div = 0.5 * torch.sum(
            p_logvar - q_logvar + (var_q + (q_mean - p_mean).pow(2)) / var_p - 1, dim=1
        )
    return kl_div.mean()  # Return mean KL divergence across batch


class HVAE(nn.Module):
    def __init__(self, img_channels=3, img_size=64, latent_dim_1=128, latent_dim_2=64):
        super(HVAE, self).__init__()
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        self.img_size = img_size

        # == Encoder q(z1|x) ==
        self.encoder_conv1 = nn.Conv2d(
            img_channels, 32, kernel_size=4, stride=2, padding=1
        )  # -> img_size/2
        self.encoder_conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # -> img_size/4
        self.encoder_conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # -> img_size/8
        self.encoder_fc = nn.Linear(128 * (img_size // 8) * (img_size // 8), 512)
        # Output mean and log-variance for z1
        self.fc_z1_mean = nn.Linear(512, latent_dim_1)
        self.fc_z1_logvar = nn.Linear(512, latent_dim_1)

        # == Encoder q(z2|z1) ==
        # Simple MLP encoder for the second level
        self.encoder_z1_to_z2 = nn.Sequential(
            nn.Linear(latent_dim_1, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        # Output mean and log-variance for z2
        self.fc_z2_mean = nn.Linear(256, latent_dim_2)
        self.fc_z2_logvar = nn.Linear(256, latent_dim_2)

        # == Decoder p(z1|z2) (Prior Network) ==
        # Maps z2 to the parameters of the prior distribution over z1
        self.prior_z2_to_z1 = nn.Sequential(
            nn.Linear(latent_dim_2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        # Output mean and log-variance for p(z1|z2)
        self.fc_prior_z1_mean = nn.Linear(256, latent_dim_1)
        self.fc_prior_z1_logvar = nn.Linear(256, latent_dim_1)

        # == Decoder p(x|z1) ==
        self.decoder_fc1 = nn.Linear(
            latent_dim_1, 128 * (img_size // 8) * (img_size // 8)
        )
        self.decoder_deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.decoder_deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        self.decoder_deconv3 = nn.ConvTranspose2d(
            32, img_channels, kernel_size=4, stride=2, padding=1
        )

    def encode(self, x):
        # q(z1|x)
        h = F.relu(self.encoder_conv1(x))
        h = F.relu(self.encoder_conv2(h))
        h = F.relu(self.encoder_conv3(h))
        h = h.view(h.size(0), -1)  # Flatten
        h = F.relu(self.encoder_fc(h))
        z1_mean = self.fc_z1_mean(h)
        z1_logvar = self.fc_z1_logvar(h)

        # q(z2|z1) - Use z1_mean as input for stability, or sample z1
        # For ELBO calculation, we need the parameters based on the *sampled* z1
        z1 = self.reparameterize(z1_mean, z1_logvar)
        h_z1 = self.encoder_z1_to_z2(z1)
        z2_mean = self.fc_z2_mean(h_z1)
        z2_logvar = self.fc_z2_logvar(h_z1)

        return z1_mean, z1_logvar, z1, z2_mean, z2_logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z1):
        # p(x|z1)
        h = F.relu(self.decoder_fc1(z1))
        h = h.view(h.size(0), 128, self.img_size // 8, self.img_size // 8)  # Reshape
        h = F.relu(self.decoder_deconv1(h))
        h = F.relu(self.decoder_deconv2(h))
        # Use sigmoid for output layer if pixel values are normalized to [0, 1]
        # Use tanh if [-1, 1], or linear if unbounded (less common for images)
        recon_x = torch.sigmoid(self.decoder_deconv3(h))
        return recon_x

    def get_prior_params(self, z2):
        # p(z1|z2)
        h_z2 = self.prior_z2_to_z1(z2)
        prior_z1_mean = self.fc_prior_z1_mean(h_z2)
        prior_z1_logvar = self.fc_prior_z1_logvar(h_z2)
        return prior_z1_mean, prior_z1_logvar

    def forward(self, x):
        # --- Inference ---
        # q(z1|x) parameters and sample
        q_z1_mean, q_z1_logvar, z1 = self.encode_z1(x)
        # q(z2|z1) parameters and sample
        q_z2_mean, q_z2_logvar = self.encode_z2(
            z1
        )  # Note: encode_z2 takes the sampled z1
        z2 = self.reparameterize(q_z2_mean, q_z2_logvar)

        # --- Generation / Prior ---
        # p(z1|z2) parameters
        p_z1_mean, p_z1_logvar = self.get_prior_params(z2)

        # --- Reconstruction ---
        # p(x|z1) - Decode using the sampled z1 from q(z1|x)
        recon_x = self.decode(z1)

        return (
            recon_x,
            q_z1_mean,
            q_z1_logvar,
            q_z2_mean,
            q_z2_logvar,
            p_z1_mean,
            p_z1_logvar,
        )

    # Separate encoding functions for clarity in forward pass
    def encode_z1(self, x):
        h = F.relu(self.encoder_conv1(x))
        h = F.relu(self.encoder_conv2(h))
        h = F.relu(self.encoder_conv3(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.encoder_fc(h))
        z1_mean = self.fc_z1_mean(h)
        z1_logvar = self.fc_z1_logvar(h)
        z1 = self.reparameterize(z1_mean, z1_logvar)
        return z1_mean, z1_logvar, z1

    def encode_z2(self, z1):
        # q(z2|z1)
        h_z1 = self.encoder_z1_to_z2(z1)
        z2_mean = self.fc_z2_mean(h_z1)
        z2_logvar = self.fc_z2_logvar(h_z1)
        return z2_mean, z2_logvar

    def sample(self, num_samples=64):
        """Generates samples from the prior p(z2) -> p(z1|z2) -> p(x|z1)"""
        with torch.no_grad():
            # Sample z2 from the standard normal prior N(0, I)
            z2_sample = torch.randn(num_samples, self.latent_dim_2).to(device)

            # Get parameters for p(z1|z2) using the sampled z2
            p_z1_mean, p_z1_logvar = self.get_prior_params(z2_sample)

            # Sample z1 from p(z1|z2)
            z1_sample = self.reparameterize(p_z1_mean, p_z1_logvar)

            # Decode z1 to generate samples in data space x
            generated_x = self.decode(z1_sample)
        return generated_x


def elbo_loss(
    recon_x,
    x,
    q_z1_mean,
    q_z1_logvar,
    q_z2_mean,
    q_z2_logvar,
    p_z1_mean,
    p_z1_logvar,
    beta=1.0,
):
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, IMAGE_SIZE * IMAGE_SIZE * 3),
        x.view(-1, IMAGE_SIZE * IMAGE_SIZE * 3),
        reduction="sum",
    ) / x.size(
        0
    )  # Average over batch

    # KL Divergence Terms
    # KL(q(z2|z1) || p(z2)) - p(z2) is the standard normal N(0, I)
    KL_z2 = kl_divergence_gaussian(q_z2_mean, q_z2_logvar)

    # KL(q(z1|x) || p(z1|z2))
    KL_z1 = kl_divergence_gaussian(q_z1_mean, q_z1_logvar, p_z1_mean, p_z1_logvar)

    # Total ELBO Loss
    # ELBO = E[log p(x|z1)] - KL(q(z1|x)||p(z1|z2)) - KL(q(z2|z1)||p(z2))
    # We maximize ELBO, which is equivalent to minimizing (-ELBO)
    loss = BCE + beta * (KL_z1 + KL_z2)

    return loss, BCE, KL_z1, KL_z2


# --- Transform the Input ---
img_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# --- Load the data ---
dataset = datasets.ImageFolder(
    root=IMAGE_PATH, transform=img_transform, target_transform=None
)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)

# --- Initialization ---
model = HVAE(
    img_channels=3,
    img_size=IMAGE_SIZE,
    latent_dim_1=LATENT_DIM_1,
    latent_dim_2=LATENT_DIM_2,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# --- Training Loop ---
def train(epoch):
    model.train()
    train_loss = 0
    bce_loss_total = 0
    kl1_loss_total = 0
    kl2_loss_total = 0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        (
            recon_batch,
            q_z1_mean,
            q_z1_logvar,
            q_z2_mean,
            q_z2_logvar,
            p_z1_mean,
            p_z1_logvar,
        ) = model(data)

        # Calculate loss
        loss, bce, kl1, kl2 = elbo_loss(
            recon_batch,
            data,
            q_z1_mean,
            q_z1_logvar,
            q_z2_mean,
            q_z2_logvar,
            p_z1_mean,
            p_z1_logvar,
            beta=BETA,
        )

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        bce_loss_total += bce.item()
        kl1_loss_total += kl1.item()
        kl2_loss_total += kl2.item()

        if batch_idx % 10 == 0:  # Print progress every 10 batches
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} "
                f"({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )

    avg_loss = train_loss / len(dataloader.dataset)
    avg_bce = bce_loss_total / len(dataloader.dataset)
    avg_kl1 = kl1_loss_total / len(dataloader.dataset)
    avg_kl2 = kl2_loss_total / len(dataloader.dataset)
    print(
        f"====> Epoch: {epoch} Average loss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | KL_z1: {avg_kl1:.4f} | KL_z2: {avg_kl2:.4f}"
    )

    # Save reconstructions and samples periodically
    if epoch % 5 == 0 or epoch == EPOCHS:
        with torch.no_grad():
            # Save generated samples
            sample_path = os.path.join(OUTPUT_PATH, "samples", f"sample_{epoch}.png")
            samples = model.sample(num_samples=64)
            save_image(samples.cpu(), sample_path, nrow=8)


if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(epoch)

    torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "hvae.pth"))
