import utils.transforms as T

class DetectionPresetTrain():
    def __init__(self, data_augmentation, size=[416, 416], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        if data_augmentation == 'hflip':
            self.transforms = T.Compose([
                T.RandomMirror(),
                # T.ToTensor(),
            ])
        elif data_augmentation == 'ssd':
            self.transforms = T.Compose([
                # T.RandomPhotometricDistort(),
                # T.RandomZoomOut(fill=list(mean)),
                # T.RandomIoUCrop(),
                # T.RandomHorizontalFlip(p=hflip_prob),
                # T.ToTensor(),
                T.ConvertFromInts(),
                T.ToAbsoluteCoords(),
                T.PhotometricDistort(),
                T.Expand(mean),
                T.RandomSampleCrop(),
                T.RandomMirror(),
                T.ToPercentCoords(),
                T.Resize(size),
                T.Normalize(mean, std)
            ])
        elif data_augmentation == 'ssdlite':
            self.transforms = T.Compose([
                # T.RandomIoUCrop(),
                # T.RandomHorizontalFlip(p=hflip_prob),
                # T.ToTensor(),
                T.RandomSampleCrop(),
                T.RandomMirror()
            ])
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, boxes=None, labels=None):
        return self.transforms(img, boxes, labels)