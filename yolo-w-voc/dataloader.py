import torch
from torch.utils.data import DataLoader
from voc import VOCDetection
from utils.augmentations import DetectionPresetTrain
from config import args

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def get_dataloader(args):
    # Dataset
    dataset = VOCDetection(img_size=args.train_size,
                        transform=DetectionPresetTrain(data_augmentation="ssd"))
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=detection_collate, num_workers=args.num_workers,
                                pin_memory=False)
    return dataloader
        
if __name__ == "__main__":
    test = next(iter(get_dataloader(args)))
    img, target = test
    img = img.cuda()
    print(img)
    # for img, target in get_dataloader(args):
    #     img = img.to('cuda')
    #     print(img)
    #     break