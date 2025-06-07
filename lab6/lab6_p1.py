import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import random
import matplotlib.pyplot as plt
import numpy as np
import sys


class FashionMNISTTrainer:
    """Guess fashion label at 28x28 image"""

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
        """Checks device, running on self.device"""
        print(f"Running on {self.device}.")

    def set_model(self):
        """Initializes CNN model with optimized architecture for Fashion-MNIST"""

        # 1. Define CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 7x7x128

        # 2. Initialize weights with Xavier uniform
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.1)

        # 3. Define network components
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.activation = nn.ReLU()

        # 4. Build sequential model
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28x32
            nn.BatchNorm2d(32),
            self.activation,
            self.pool,  # 14x14x32
            nn.Conv2d(32, 64, 3, padding=1),  # 14x14x64
            nn.BatchNorm2d(64),
            self.activation,
            self.pool,  # 7x7x64
            nn.Conv2d(64, 128, 3, padding=1),  # 7x7x128
            nn.BatchNorm2d(128),
            self.activation,
            self.pool,  # 3x3x128
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            self.activation,
            self.dropout,
            nn.Linear(256, 10),
        ).to(self.device)

        # 5. Configure loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-4
        )

    def train(self):
        """Enhanced training loop with learning rate scheduling"""
        self.model.train()

        epoch_num = 20

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.005,
            steps_per_epoch=len(self.train_data_loader),
            epochs=epoch_num,
            pct_start=0.3,
            anneal_strategy="cos",
        )

        for epoch in range(epoch_num):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(self.train_data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward-backward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                scheduler.step()

                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Epoch statistics
            epoch_acc = 100 * correct / total
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch+1}/{epoch_num} \t Loss: {running_loss/len(self.train_data_loader):.4f} Acc: {epoch_acc:.2f}% \t LR: {current_lr:.6f}"
            )

    @torch.no_grad()
    def eval(self, image):
        """Enhanced evaluation with input validation"""
        self.model.eval()

        # Input dimension handling
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # Add channel & batch dim
        elif image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim

        image = image.to(self.device)
        outputs = self.model(image)
        return torch.argmax(outputs).item()

    def eval_all(self):
        """evaluate all images which are in test data"""
        acc_cnt, tot_cnt = 0, 0
        for image, label in self.test_data:
            pred = self.eval(image)
            acc_cnt += label == pred
            tot_cnt += 1

            # 진행률 표시
            progress = (tot_cnt + 1) / len(self.test_data)
            bar_length = 40
            bar = "#" * int(bar_length * progress) + "-" * (
                bar_length - int(bar_length * progress)
            )
            sys.stdout.write(
                f"\r|{bar}| {progress*100:.1f}% ({tot_cnt+1}/{len(self.test_data)})"
            )
            sys.stdout.flush()

        print(f"\nAccuracy: {acc_cnt / tot_cnt * 100:.2f}%")

    def sample_test_image(self):
        """test with a random sample image"""
        r = self.random.integers(low=0, high=len(self.test_data))
        return self.test_data[r]

    def export_model(self, file_name="lab6_p1.pth"):
        """export model to a .pth file"""
        torch.save(self.model.state_dict(), file_name)

    def import_model(self, file_name="lab6_p1.pth"):
        """import model by reading .pth file"""
        state_dict = torch.load(file_name, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict)

    def set_seed(self, seed=2025):
        """set seed, fixe seed at 2025"""
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
