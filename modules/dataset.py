class Dataset:
    def __init__(self, name):
        self.name = name

    def summary(self):
        ...

    def download(self):
        ...

    def preprocess(self):
        ...


class ImageNet(Dataset):
    def __init__(self):
        super().__init__('ImageNet')
        self.num_classes = 1000

    def preprocess(self):
        print("Resizing to 224x224 and normalizing using ImageNet mean/std.")

    def summary(self):
        return f"{super().summary()}, Classes: {self.num_classes}"