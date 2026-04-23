import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataset(batch_size):
    image_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=image_transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=image_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
