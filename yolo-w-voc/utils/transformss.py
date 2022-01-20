# taken directly from torchvision's github: https://github.com/pytorch/vision/blob/main/references/detection/transforms.py

from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, boxes, labels):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
        return image, boxes, labels


class ToTensor(nn.Module):
    def forward(self, image, boxes, labels):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, boxes, labels


class PILToTensor(nn.Module):
    def forward(self, image, boxes, labels):
        image = F.pil_to_tensor(image)
        return image, boxes, labels


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image, boxes, labels):
        image = F.convert_image_dtype(image, self.dtype)
        return image, boxes, labels


class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(self, image, boxes, labels):
        if boxes is None or labels is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        orig_w, orig_h = F._get_image_size(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, boxes, labels

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
                cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                # test
                # left, top, right, bottom = [torch.Tensor(coord).item() for coord in [left, top, right, bottom]]
                # end test

                box = boxes[is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    # box, torch.tensor([[left, top, right, bottom]], dtype=box.dtype, device=box.device)
                    torch.tensor(box), torch.tensor([[left, top, right, bottom]])
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                boxes = box
                boxes = torch.tensor(boxes)
                labels = labels[is_within_crop_area]
                boxes[:, 0::2] -= left
                boxes[:, 1::2] -= top
                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, new_w)
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, new_h)
                # boxes[:, 0::2].clamp_(min=0, max=new_w)
                # boxes[:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, boxes, labels


class RandomZoomOut(nn.Module):
    def __init__(
        self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5
    ):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(self, image, boxes, labels):
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) < self.p:
            return image, boxes, labels

        orig_w, orig_h = F._get_image_size(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h) :, :] = image[
                ..., :, (left + orig_w) :
            ] = v

        if boxes is not None:
            boxes[:, 0::2] += left
            boxes[:, 1::2] += top

        return image, boxes, labels


class RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float] = (0.5, 1.5),
        saturation: Tuple[float] = (0.5, 1.5),
        hue: Tuple[float] = (-0.05, 0.05),
        brightness: Tuple[float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(self, image, boxes, labels):
        # if isinstance(image, torch.Tensor):
        #     if image.ndimension() not in {2, 3}:
        #         raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
        #     elif image.ndimension() == 2:
        #         image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels = F._get_image_num_channels(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, boxes, labels