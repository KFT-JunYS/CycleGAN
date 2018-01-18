import os
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ImageFolder(data.Dataset):
    """Custom Dataset 의 호환성을 위해 구성
       보통의 경우 torchvision.datasets.ImageFolder 를 사용하면 됨."""

    def __init__(self, root, transform=None):
        """이미지 경로와 이미지 처리를 위한 초기화 작업"""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):
        """이미지 경로에서 이미지를 RGB 형태로 읽어와서 transform 에 구성한 내용대로 이미지를 만들어 리턴"""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """이미지 파일의 총 개수 리턴"""
        return len(self.image_paths)


def get_loader(config, num_workers=2):
    """transform 구성을 통해 어떠한 방식으로 이미지를 읽어올지 설정하고 처리한 결과를 리턴"""
    transform = transforms.Compose([
        transforms.Scale(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    A_train = ImageFolder(config.dataset + config.train_subfolder + 'A', transform)
    B_train = ImageFolder(config.dataset + config.train_subfolder + 'B', transform)
    A_test  = ImageFolder(config.dataset + config.test_subfolder + 'A', transform)
    B_test  = ImageFolder(config.dataset + config.test_subfolder + 'B', transform)

    A_train_loader = data.DataLoader(dataset=A_train,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)

    B_train_loader = data.DataLoader(dataset=B_train,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     num_workers=num_workers)

    A_test_loader = data.DataLoader(dataset=A_test,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

    B_test_loader = data.DataLoader(dataset=B_test,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

    return A_train_loader, B_train_loader, A_test_loader, B_test_loader