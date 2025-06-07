import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np


class FashionMNISTTrainer:
	def __init__(self, seed=2025):
		self.set_seed(seed=seed)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.transform = transforms.Compose([
			transforms.ToTensor(),
		])

		# load dataset
		self.train_data = datasets.FashionMNIST(
			root='FashionMNIST_data/',
			train=True,
			transform=self.transform,
			download=True
		)
		self.test_data = datasets.FashionMNIST(
			root='FashionMNIST_data/',
			train=False,
			transform=self.transform,
			download=True
		)

		self.train_data_loader = DataLoader(
			dataset=self.train_data,
			batch_size=100,
			shuffle=True,	# Shuffle the order of dataset
			drop_last=True
		)

		# model used for training
		self.set_model()

	def check_device(self):
		print(f"Running on {self.device}.")

	def set_model(self):
		"""
		write your docstring...
		"""

		### write your code...
		# 1. set network
		# ?
		
		# 2. initialize weight (if needed)
		# ?
		
		# 3. set activation function
		# ?
		
		# 4. set model
		self.model = nn.Sequential(
			# ?
		).to(self.device)

		# 5. set loss and optimizer
		self.criterion = None	# ?
		self.optimizer = None	# ?
	
	def train(self):
		"""
		write your docstring...
		"""

		# train your model here
		# you may print the loss of your model

		# if you are running this code on 'cuda', (check by calling check_device())
		# you have to pass your image to device by using '.to(self.device)'
		# the same applies to function eval(...):

		# set to train mode
		self.model.train()

		# ?

	@torch.no_grad()
	def eval(self, image):
		"""
		write your docstring...
		"""
		
		# image will have size of 28 * 28
		# you should return the label predicted by your model
		# this function only predicts the label of a single image

		# set model to eval mode
		self.model.eval()

		# ?

		return predicted_label

	def eval_all(self):
		acc_cnt, tot_cnt = 0, 0
		for image, label in self.test_data:
			pred = self.eval(image)
			acc_cnt += label == pred
			tot_cnt += 1

		print(f"Accuracy: {acc_cnt / tot_cnt * 100:.2f}%")

	def sample_test_image(self):
		r = self.random.integers(low=0, high=len(self.test_data))
		return self.test_data[r]

	def export_model(self, file_name="lab6_p1.pth"):
		torch.save(self.model.state_dict(), file_name)

	def import_model(self, file_name="lab6_p1.pth"):
		state_dict = torch.load(file_name, map_location=torch.device(self.device))
		self.model.load_state_dict(state_dict)

	def set_seed(self, seed=2025):
		self.random = np.random.default_rng(seed)


if __name__ == "__main__":
	# 1. generate trainer
	trainer = FashionMNISTTrainer()

	# 2. run train
	trainer.train()

	# 3. evaluate on test set
	trainer.eval_all()

	# 4. pick a image and pass through the network
	image, label = trainer.sample_test_image()
	print(f'Label: {label}')
	print(f'Prediction: {trainer.eval(image)}')

	# 5. export model
	trainer.export_model()

	# 6. import model
	trainer.import_model()

	# 7. evaluate on test set
	# Note: ACCURACY MUST REMAIN THE SAME AS 3.
	trainer.eval_all()