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


def latent_space(model, dataloader, config, legend=True, sample="mean"):
    """display how the latent space clusters different data points in vMF space"""

    model.eval()
    x = torch.rand((1, model.z_dim))
    y = torch.rand(1)

    for data in dataloader:  # calculate the latent codes
        images = data[0].to(config.device)
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
        y = torch.cat((y, data[-1]), 0) 

    x = x.detach().numpy()[1:, :]
    y = y.detach().numpy()[1:]

    if model.z_dim == 3:
        ax = plt.figure(figsize=(10, 10)).add_subplot(111, projection='3d')
        for key, value in config.label_list.items():
            ax.scatter(x[np.where(y == int(key)), 0],
                       x[np.where(y == int(key)), 1],
                       x[np.where(y == int(key)), 2],
                       alpha=0.8, label=value)

    elif model.z_dim == 2:
        fig, ax = plt.subplots()
        for key, value in config.label_list.items():
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
    ax.set_xlim(-1 * config.radius, config.radius)
    ax.set_ylim(-1 * config.radius, config.radius)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)

    if model.z_dim == 3:
        ax.set_zlim(-1 * config.radius, config.radius)
        ax.set_zlabel("Z", fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_aspect('equal')
    plt.show()


def manifold_sphere(model, config, theta=(0,), z_resolution=10):
    """
    theta: the arc between intersecting lines - one is (0, 1) on the x-y plane, ranged in 0-2pi;
    z_resolution: how many pics to be shown in a 1/4 arc
    """
    model.eval()

    if model.z_dim == 3:
        fig, ax = plt.subplots(len(theta), z_resolution)
        plt.setp(ax.flat, xticks=[], yticks=[])

        for j, arc in enumerate(theta):
            grid = torch.tensor([[0, 0, 0]])

            for i in range(z_resolution):

                phi = pi*(-0.5) + (i+1)*pi/z_resolution
                sample = torch.tensor([[np.sin(arc)*np.cos(phi), np.cos(arc)*np.cos(phi), np.sin(phi)]]) * config.radius
                grid = torch.cat((grid, sample), 0)

            grid = grid[1:, :].type("torch.FloatTensor").to(config.device)
            img = model.decoder(grid)
            img = img.cpu().detach().numpy()

            for i in range(z_resolution):
                ax[j][i].imshow(img[i, 0, :, :])

        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)

    elif model.z_dim == 2:
        fig, ax = plt.subplots(1, len(theta))
        plt.setp(ax.flat, xticks=[], yticks=[])

        grid = torch.tensor([[0, 0]])

        for arc in theta:

            sample = torch.tensor([[np.sin(arc), np.cos(arc)]]) * config.radius
            grid = torch.cat((grid, sample), 0)

        grid = grid[1:, :].type("torch.FloatTensor").to(config.device)
        img = model.decoder(grid)
        img = img.cpu().detach().numpy()

        for i in range(len(theta)):
            ax[i].imshow(img[i, 0, :, :])

    else: 
        raise NotImplemented

    plt.show()


def reconstruction(model, dataloader, n_show, config):
    # random visualization of the reconstruction
    model.eval()

    rand_idx = np.random.randint(0, len(dataloader), 1)

    for idx, data in enumerate(dataloader):  # calculate the latent codes
        if idx == rand_idx:

            if config.aug:
                if config.Ncrop:
                    _, _, c, h, w = data[1].shape
                    images = data[1].to(config.device).view(-1, c, h, w)
                else:
                    images = data[1].to(config.device)
            elif not config.aug:
                images = data[0].to(config.device)

            _, _, _, (img_rec, _) = model(images)

            f, a = plt.subplots(2, n_show)
            plt.setp(a.flat, xticks=[], yticks=[])

            for i in range(n_show):
                a[0][i].imshow(images.cpu().numpy()[i, 0, :, :])
                a[1][i].imshow(img_rec.cpu().detach().numpy()[i, 0, :, :])

            f.subplots_adjust(wspace=0, hspace=0, bottom=0.15, right=0.88)
            plt.show()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='earlystopping_checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.measure = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, measure, direction, model):

        if direction == "min":
            score = -measure
        elif direction == "max":
            score = measure

        if self.best_score is None:
            self.best_score = score
            self.save_all(measure, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_all(measure, model)
            self.counter = 0

    def save_checkpoint(self, measure, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation measure ({self.measure:.6f} --> {measure:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.measure = measure

    def save_all(self, measure, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation measure ({self.measure:.6f} --> {measure:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.measure = measure


