# single epoch train function
import torch
from torch.cuda.amp import GradScaler, autocast  # mixed precision
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, optimizer, device, scaler, w_cla, augment=False):

	model.train()  # or model.eval(), to activate or freeze the BN and dropout layers

	Loss_R, Loss_K, Loss_C = 0, 0, 0

	for data in tqdm(train_loader, ncols=100, desc="Train:"): 

		images, labels = data
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()

		with autocast():

            if augment:
                bs, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)  # the first dimension becomes Ncrop x batch-size
                labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
            
			_, (q_z, p_z), _, (images_, labels_) = model(images)  
			# output of encoder (prior and posterior distribution) and decoder

			loss_recon = nn.MSELoss(reduction='none')(images_, images).sum(-1).mean()

			if model.distribution == 'normal':
				loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
			elif model.distribution == 'vmf':
				loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
			else:
				raise NotImplemented

			loss_cla = nn.CrossEntropyLoss(reduction="none")(labels_, labels).sum(-1).mean()

			if loss_cla > 100:
				loss = loss_recon + loss_KL + loss_cla * w_cla
			else:
				loss = loss_recon + loss_KL  # just need a moderate classification accuracy

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		Loss_R += loss_recon
		Loss_K += loss_KL
		Loss_C += loss_cla

	return Loss_R, Loss_K, Loss_C
