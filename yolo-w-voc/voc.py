import os.path as osp
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from utils.augmentations import DetectionPresetTrain
from PIL import Image

path_to_dir = osp.dirname(osp.abspath("__file__"))
VOC_ROOT = path_to_dir + "/VOC/"
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# Transform VOC annotation into Tensor of bbox coords and label index
class VOCAnnotationTransform():
    def __init__(self):
        self.class_to_idx = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    
    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        result = []

        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1 # why -1???
                # normalize w.r.t width and height
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            
            label_idx = self.class_to_idx[name]
            bndbox.append(label_idx)
            result += [bndbox] # [xmin, ymin, xmax, ymax, label_idx]
        
        return result # [[xmin, ymin, xmax, ymax, label_idx], [...],...]

# Dataset
class VOCDetection(Dataset):
    """VOC Detection Dataset Object

    Arguments:
        root (string): filepath to the VOC folder.
        img_sets (string): image sets to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, img_size, root=VOC_ROOT,
            img_sets=[('2007', 'trainval'), ('2012', 'trainval')],
            transform=None, target_transform=VOCAnnotationTransform(),
            dataset_name='VOC0712'):
        self.root = root
        self.img_size = img_size
        self.img_set = img_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._anno_path = osp.join('%s', 'Annotations', '%s.xml')
        self._img_path = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list() # list of ids of imgs
        for (year, name) in img_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
    
    def __getitem__(self, index):
        im, gt = self.pull_item(index) # image, ground truth (boxes and labels)

        return im, gt
    
    def __len__(self):
        return len(self.ids)            
    
    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._anno_path % img_id).getroot()
        img = cv2.imread(self._img_path % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb bc cv2
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return torch.from_numpy(img).permute(2, 0, 1), target
        
if __name__ == "__main__":
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels

    
    img_size = 640
    # dataset
    # dataset = VOCDetection(VOC_ROOT, img_size, [('2007', 'trainval')],
    #                         DetectionPresetTrain(data_augmentation="ssd"),
    #                         # BaseTransform([img_size, img_size], (0, 0, 0)),
    #                         VOCAnnotationTransform())
    dataset_no_transform = VOCDetection(img_size, img_sets=[('2007', 'trainval')],
                            # DetectionPresetTrain(data_augmentation="ssd"),
                            transform=BaseTransform([img_size, img_size], (0, 0, 0)),
                            target_transform=VOCAnnotationTransform())
    dataset_yes_transform = VOCDetection(img_size, img_sets=[('2007', 'trainval')],
                            transform=DetectionPresetTrain(data_augmentation="ssd"),
                            # BaseTransform([img_size, img_size], (0, 0, 0)),
                            target_transform=VOCAnnotationTransform())
    # test
    i = 10
    im, gt = dataset_no_transform.pull_item(i)
    img = im.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()

    for box in gt:
        xmin, ymin, xmax, ymax, label = box
        xmin *= img_size
        ymin *= img_size
        xmax *= img_size
        ymax *= img_size
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imwrite('-1.jpg', img)
    img = cv2.imread('-1.jpg')

    im2, gt2 = dataset_yes_transform.pull_item(i)
    img2 = im2.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()
    height, width, _ = img2.shape

    for box2 in gt2:
        xmin2, ymin2, xmax2, ymax2, label2 = box2
        xmin2 *= width
        ymin2 *= height
        xmax2 *= width
        ymax2 *= height
        img2 = cv2.rectangle(img2, (int(xmin2), int(ymin2)), (int(xmax2), int(ymax2)), (0, 0, 255), 2)

    cv2.imwrite('-2.jpg', img2)
    img2 = cv2.imread('-2.jpg')
    # for i in range(10):
    #     im, gt = dataset.pull_item(i)
    #     img = im.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8).copy()

    #     for box in gt:
    #         xmin, ymin, xmax, ymax, label = box
    #         xmin *= img_size
    #         ymin *= img_size
    #         xmax *= img_size
    #         ymax *= img_size
    #         img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    #     cv2.imwrite('-1.jpg', img)
    #     img = cv2.imread('-1.jpg')

        # cv2.imshow('gt', img)
        # cv2.waitKey(0)
