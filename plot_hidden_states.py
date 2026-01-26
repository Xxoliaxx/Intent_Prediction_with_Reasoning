import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- LOAD SAVED STATES ----
states = torch.load(
    "checkpoints/User-trajectory-hrm ACT-torch/<run_name>/hidden_states/hrm_hidden_states.pt",
    map_location="cpu"
)

zH = states["z_H"]  # [T, seq_len, hidden]
zL = states["z_L"]

T, S, D = zH.shape
print(f"Loaded z_H: {zH.shape}, z_L: {zL.shape}")

# ============================================================
# FIGURE 1: PCA TRAJECTORY (H-level, CLS token)
# ============================================================

X = zH[:, 0]  # token 0 across ACT steps → [T, D]

pca = PCA(n_components=2)
Y = pca.fit_transform(X)

plt.figure(figsize=(4, 4))
plt.plot(Y[:, 0], Y[:, 1], marker="o")
plt.title("H-level hidden-state trajectory (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_h_trajectory.png")
plt.close()

# ============================================================
# FIGURE 2: NORM DECAY ACROSS ACT STEPS
# ============================================================

h_norm = zH.norm(dim=-1).mean(dim=-1)
l_norm = zL.norm(dim=-1).mean(dim=-1)

plt.figure(figsize=(5, 3))
plt.plot(h_norm, label="HRM-H")
plt.plot(l_norm, label="HRM-L")
plt.xlabel("ACT step")
plt.ylabel("||z||")
plt.legend()
plt.tight_layout()
plt.savefig("norm_decay.png")
plt.close()

# ============================================================
# FIGURE 3: TOKEN HEATMAP (FINAL STEP)
# ============================================================

plt.figure(figsize=(6, 3))
plt.imshow(zH[-1].norm(dim=-1).unsqueeze(0), aspect="auto")
plt.colorbar(label="||z||")
plt.yticks([])
plt.xlabel("Token index")
plt.title("Final H-level token activations")
plt.tight_layout()
plt.savefig("token_heatmap.png")
plt.close()

print("Figures saved.")
