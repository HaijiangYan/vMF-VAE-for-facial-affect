# utils for vMF-VAE modelling
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def latent_space(model, dataloader, latent_dims=3, legend=True, label_list=None, device="cpu", sample="mean"):
    """display how the latent space clusters different data points in vMF space"""

    model.eval()
    x = torch.rand((1, latent_dims))
    y = torch.rand(1)

    for images, labels in dataloader:  # calculate the latent codes
        images = images.to(device)
        latent_code, _, z, _ = model(images)

        # plot with mean values or random samples from posterior distributions
        if sample == "mean":
            latent_embeddings, _ = latent_code
            latent_embeddings = latent_embeddings.to("cpu")
        elif sample == "random":
            latent_embeddings = z.to("cpu")
        else:
            raise NotImplemented

        x = torch.cat((x, latent_embeddings), 0) 
        y = torch.cat((y, labels), 0) 

    x = x.detach().numpy()[1:, :]
    y = y.detach().numpy()[1:]

    if latent_dims == 3:
        ax = plt.figure(figsize=(10, 10)).add_subplot(111, projection='3d')
        for key, value in label_list.items():
            ax.scatter(x[np.where(y == int(key)), 0],
                       x[np.where(y == int(key)), 1],
                       x[np.where(y == int(key)), 2],
                       alpha=0.8, label=value)

    elif latent_dims == 2:
        fig, ax = plt.subplots()
        for key, value in label_list.items():
            ax.scatter(x[np.where(y == int(key)), 0],
                       x[np.where(y == int(key)), 1],
                       alpha=0.8, label=value)
    else: 
        raise NotImplemented
    print("Number of data points:", len(x))

    # plt.title("Data Points in Latent Space")
    if legend == True:
        ax.legend(fontsize=12)
    # plt.savefig('./figure/save.jpg')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()


def manifold(model, theta=(0,), resolution=10, device="cpu"):
    """
    theta: the arc between intersecting lines - one is (0, 1) on the x-y plane, ranged in 0-2pi;
    resolution: how many pics to be shown in a 1/4 arc
    """
    model.eval()

    fig, ax = plt.subplots(len(theta), resolution)
    plt.setp(ax.flat, xticks=[], yticks=[])

    for j, arc in enumerate(theta):
        grid = torch.tensor([[0, 0, 0]])

        for i in range(resolution):

            phi = pi*(-0.5) + (i+1)*pi/resolution
            sample = torch.tensor([[np.sin(arc)*np.cos(phi), np.cos(arc)*np.cos(phi), np.sin(phi)]])
            grid = torch.cat((grid, sample), 0)

        grid = grid[1:, :].type("torch.FloatTensor").to(device)
        img = model.decoder(grid)
        img = img.cpu().detach().numpy()

        for i in range(resolution):
            ax[j][i].imshow(img[i, 0, :, :])

    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)
    plt.show()


def reconstruction(model, dataloader, n_show, device="cpu"):
    # random visualization of the reconstruction
    model.eval()

    rand_idx = np.random.randint(0, len(dataloader), 1)

    for idx, (images, labels) in enumerate(dataloader):  # calculate the latent codes
        if idx == rand_idx:
            images = images.to(device)
            _, _, _, (img_rec, _) = model(images)

            f, a = plt.subplots(2, n_show)
            plt.setp(a.flat, xticks=[], yticks=[])

            for i in range(n_show):
                a[0][i].imshow(images.cpu().numpy()[i, 0, :, :])
                a[1][i].imshow(img_rec.cpu().detach().numpy()[i, 0, :, :])

            f.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)
            plt.show()





