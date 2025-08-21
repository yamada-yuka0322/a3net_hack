# pytorch_vae_basic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- VAE 本体 ---------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, in_dim=28*28, hidden_dim=400, z_dim=20):
        super().__init__()
        # Encoder: x -> hidden -> (mu, logvar)
        self.enc_fc = nn.Linear(in_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_logvar = nn.Linear(hidden_dim, z_dim)

        # Decoder: z -> hidden -> x_recon(logits)
        self.dec_fc1 = nn.Linear(z_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, in_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc(x))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps,  sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        logits = self.dec_fc2(h)  # シグモイドは BCEWithLogitsLoss に内包させる
        return logits

    def forward(self, x):
        # x: (B, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

# ---- VAE Loss（再構成 + KL） -----------------------------------------------
def vae_loss(recon_logits, x, mu, logvar, reduction="sum"):
    # 再構成誤差：Bernoulli仮定 -> BCE with logits（数値安定）
    # x は [0,1] の画素を想定
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_logits, x, reduction=reduction
    )
    # KL( q(z|x) || N(0, I) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl

# ---- 使い方（学習ループの最小例） -------------------------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ（MNIST 0-1 正規化 & Tensor）
    transform = transforms.Compose([
        transforms.ToTensor(),                # (B,1,28,28) 0-1
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    model = VAE(in_dim=28*28, hidden_dim=400, z_dim=20).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, 11):
        total, total_recon, total_kl = 0.0, 0.0, 0.0
        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)  # flatten to (B, 784)
            opt.zero_grad()
            recon_logits, mu, logvar = model(x)
            loss, recon_loss, kl = vae_loss(recon_logits, x, mu, logvar, reduction="sum")
            loss.backward()
            opt.step()

            total += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()

        n = len(train_loader.dataset)
        print(f"Epoch {epoch:02d} | "
              f"Loss {(total/n):.4f} | Recon {(total_recon/n):.4f} | KL {(total_kl/n):.4f}")

    # サンプル生成（学習後）
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, 20).to(device)
        logits = model.decode(z)
        samples = torch.sigmoid(logits).cpu()  # [0,1] 画像
        # 例：最初の1枚を保存
        from torchvision.utils import save_image
        save_image(samples.view(64, 1, 28, 28), "vae_samples.png", nrow=8)
        print("Saved: vae_samples.png")
