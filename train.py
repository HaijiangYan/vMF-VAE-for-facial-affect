# single epoch train function
import torch
from torch.cuda.amp import GradScaler, autocast  # mixed precision
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def train(model, train_loader, optimizer, scaler, config):

	model.train()  # or model.eval(), to activate or freeze the BN and dropout layers

	Loss_R, Loss_K, Loss_C = 0, 0, 0

	# for data in tqdm(train_loader, ncols=100, desc="Train:"): 
	for data in train_loader: 

		if not config.aug:
			base_images, labels, _ = data
			base_images, labels= base_images.to(config.device), labels.to(config.device)
		elif config.aug:
			base_images, aug_images, labels, _ = data
			base_images, aug_images, labels= base_images.to(config.device), aug_images.to(config.device), labels.to(config.device)
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
				(embeddings, _), (q_z, p_z), _, (images_, labels_) = model(aug_images)  

			elif not config.aug:
				(embeddings, _), (q_z, p_z), _, (images_, labels_) = model(base_images)  
			# output of encoder (prior and posterior distribution) and decoder

			loss_recon = config.loss(reduction='none')(images_, base_images).sum(-1).mean()

			if model.distribution == 'normal':
				loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
			elif model.distribution == 'vmf':
				loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
			else:
				raise NotImplemented

			loss_cla = nn.CrossEntropyLoss(reduction="none")(labels_, labels).sum(-1).mean()

			# regularization of within-cluster dispersion
			# if config.regularization: 
			# 	all_var = torch.ones(len(torch.unique(labels)))
			# 	all_var = all_var.to(config.device)

			# 	for i, label in enumerate(torch.unique(labels)):
			# 		cluster = embeddings[torch.where(labels==label)[0], :]
			# 		all_var[i] = torch.mean(torch.var(cluster, dim=0, unbiased=True))

			# 	all_var = torch.var(all_var, dim=0, unbiased=True)


			# 	# if loss_cla > 112:
			# 	# 	loss = loss_recon + loss_KL + loss_cla * config.weight_class
			# 	# else:
			# 	# 	loss = loss_recon + loss_KL  # just need a moderate classification accuracy
			# 	loss = loss_recon + loss_KL + loss_cla * config.weight_class + all_var * config.weight_regularization
			# else:
			# 	loss = loss_recon + loss_KL + loss_cla * config.weight_class
			loss = loss_recon + loss_KL + loss_cla * config.weight_class

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		Loss_R += loss_recon
		Loss_K += loss_KL
		Loss_C += loss_cla

	return Loss_R, Loss_K, Loss_C
