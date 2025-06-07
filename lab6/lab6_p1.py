import torch
from torch import nn
import torch.optim.adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import numpy as np


class FashionMNISTTrainer:
    def __init__(self, seed=2025):
        """Initialize fashion MNIST trainer
        set seed, load dataset, set model"""
        self.set_seed(seed=seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # load dataset
        self.train_data = datasets.FashionMNIST(
            root="FashionMNIST_data/",
            train=True,
            transform=self.transform,
            download=True,
        )
        self.test_data = datasets.FashionMNIST(
            root="FashionMNIST_data/",
            train=False,
            transform=self.transform,
            download=True,
        )

        self.train_data_loader = DataLoader(
            dataset=self.train_data,
            batch_size=100,
            shuffle=True,  # Shuffle the order of dataset
            drop_last=True,
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
        linear1 = nn.Linear(784, 512, bias=True)
        linear2 = nn.Linear(512, 512, bias=True)
        linear3 = nn.Linear(512, 10, bias=True)

        # 2. initialize weight (if needed)
        nn.init.xavier_uniform(linear1.weight)
        nn.init.xavier_uniform(linear2.weight)
        nn.init.xavier_uniform(linear3.weight)

        # 3. set activation function
        act_func = nn.ReLU()

        # 4. set model
        self.model = nn.Sequential(linear1, act_func, linear2, act_func, linear3).to(
            self.device
        )

        # 5. set loss and optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

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

        batch_num = len(self.train_data_loader)
        epoch_num = 20

        cost_log = []
        for epoch in range(epoch_num):
            avg_cost = 0

            for X, Y in self.train_data_loader:  # X is image, Y is label (0~9)
                X = X.view(-1, 28 * 28).to(self.device)
                Y = Y.to(self.device)
                self.optimizer.zero_grad()  # Initialize gradients to 0
                guess = self.model(X)  # Model's guess
                cost = self.criterion(guess, Y)  # Loss
                cost.backward()  # Compute gradient
                self.optimizer.step()  # Update weights
                avg_cost += cost / batch_num

            cost_log.append(float(avg_cost))
            print(f"Epoch: {epoch+1:02d}, cost={avg_cost:.9f}")

        plt.plot(np.arange(epoch_num), cost_log)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()
        print("Model training complete!")

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

        X_test = self.test_data.test_data.view(-1, 28 * 28).float().to(self.device)
        Y_test = self.test_data.test_labels.to(self.device)

        prediction = self.model(X_test)  # Pass 10k images and predict their label
        correct_pred = torch.argmax(prediction, 1) == Y_test  # True if correct
        accuracy = correct_pred.float().mean()

        # Select random MNIST image and test on it
        r = random.randint(0, len(self.test_data) - 1)
        X_single_data = (
            self.test_data.test_data[r : r + 1]
            .view(-1, 28 * 28)
            .float()
            .to(self.device)
        )
        Y_single_data = self.test_data.test_labels[r : r + 1].to(self.device)
        single_pred = self.model(X_single_data)

        predicted_label = torch.argmax(single_pred, 1).item()

        return predicted_label

    def eval_all(self):
        acc_cnt, tot_cnt = 0, 0
        for image, label in self.test_data:
            pred = self.eval(image)
            acc_cnt += label == pred
            tot_cnt += 1
            print(f"cnt: {tot_cnt} / {len(self.test_data)}")

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
    print(f"Label: {label}")
    print(f"Prediction: {trainer.eval(image)}")

    # 5. export model
    trainer.export_model()

    # 6. import model
    trainer.import_model()

    # 7. evaluate on test set
    # Note: ACCURACY MUST REMAIN THE SAME AS 3.
    trainer.eval_all()
