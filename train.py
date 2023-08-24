# single epoch train function
import torch
from torch.cuda.amp import GradScaler, autocast  # mixed precision
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, optimizer, scaler, config):

	model.train()  # or model.eval(), to activate or freeze the BN and dropout layers

	Loss_R, Loss_K, Loss_C = 0, 0, 0

	# for data in tqdm(train_loader, ncols=100, desc="Train:"): 
	for data in train_loader: 

		if not config.aug:
			base_images, labels = data
			base_images, labels = base_images.to(config.device), labels.to(config.device)
		elif config.aug:
			base_images, aug_images, labels = data
			base_images, aug_images, labels = base_images.to(config.device), aug_images.to(config.device), labels.to(config.device)
		else:
			raise NotImplemented

		optimizer.zero_grad()

		with autocast():

			if config.aug:
				if config.Ncrop:
					bs, ncrops, c, h, w = aug_images.shape
					aug_images = aug_images.view(-1, c, h, w)  # the first dimension becomes Ncrop x batch-size
					base_images = torch.repeat_interleave(base_images, repeats=ncrops, dim=0)
					labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
				_, (q_z, p_z), _, (images_, labels_) = model(aug_images)  

			elif not config.aug:
				_, (q_z, p_z), _, (images_, labels_) = model(base_images)  
			# output of encoder (prior and posterior distribution) and decoder

			loss_recon = nn.MSELoss(reduction='none')(images_, base_images).sum(-1).mean()

			if model.distribution == 'normal':
				loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
			elif model.distribution == 'vmf':
				loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
			else:
				raise NotImplemented

			loss_cla = nn.CrossEntropyLoss(reduction="none")(labels_, labels).sum(-1).mean()

			if loss_cla > 105:
				loss = loss_recon + loss_KL + loss_cla * config.weight_class
			else:
				loss = loss_recon + loss_KL  # just need a moderate classification accuracy
			# loss = loss_recon + loss_KL + loss_cla * config.weight_class

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		Loss_R += loss_recon
		Loss_K += loss_KL
		Loss_C += loss_cla

	return Loss_R, Loss_K, Loss_C
