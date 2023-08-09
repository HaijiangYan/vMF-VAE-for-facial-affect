# utils for vMF-VAE modelling
import torch
import numpy as np
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def latent_space(model, dataloader, latent_dims=3, legend=True, device="cpu"):
    """display how the latent space clusters different data points in vMF space"""

    model.eval()
    x = torch.rand((1, latent_dims))
    y = torch.rand(1)

    for images, labels in dataloader:  # calculate the latent codes
        images = images.to(device)
        latent_code, _, _, _ = model(images)
        latent_embeddings, _ = latent_code
        latent_embeddings = latent_embeddings.to("cpu")
        x = torch.cat((x, latent_embeddings), 0) 
        y = torch.cat((y, labels), 0) 

    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')

    x = x.detach().numpy()[1:, :]
    y = y.detach().numpy()[1:]

    ax.scatter(x[:, 0], x[:, 1], x[:, 2], alpha=0.5, c=y)

    # plt.title("Data Points in Latent Space")
    if legend == True:
        ax.legend(fontsize=19)
    # plt.savefig('./figure/save.jpg')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel("X", fontsize=17)
    ax.set_ylabel("Y", fontsize=17)
    ax.set_zlabel("Z", fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.show()




