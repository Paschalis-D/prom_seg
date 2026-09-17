from pathlib import Path
from typing import Optional
import json

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, functional
from pycocotools import mask as mask_utils


class FilamentDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        json_path: Optional[Path] = None,
        is_train: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.is_train = is_train
        self.to_tensor = ToTensor()

        if is_train and json_path is None:
            raise ValueError("json_path is required when is_train=True")

        if is_train:
            with open(json_path, "r") as f:
                coco = json.load(f)

            # image_id -> (file_name, height, width)
            self.image_meta: dict[str, tuple[str, int, int]] = {
                img["id"]: (img["file_name"], img["height"], img["width"])
                for img in coco["images"]
            }

            # image_id -> list of polygon segmentations (one per filament)
            self.id_to_segmentations: dict[str, list] = {}
            for ann in coco["annotations"]:
                self.id_to_segmentations.setdefault(ann["image_id"], []).append(
                    ann["segmentation"]
                )

            # Stable, precomputed index -> image_id mapping
            self.ids: list[str] = sorted(self.image_meta.keys())
        else:
            self.image_meta = {}
            self.id_to_segmentations = {}
            self.ids = sorted(p.name for p in self.images_dir.iterdir()
                              if p.suffix.lower() in {".jpg", ".jpeg"})

    def __len__(self) -> int:
        return len(self.ids)

    def _build_mask(self, image_id: str, height: int, width: int) -> torch.Tensor:
        canvas = np.zeros((height, width), dtype=np.uint8)
        for polygon in self.id_to_segmentations.get(image_id, []):
            rles = mask_utils.frPyObjects(polygon, height, width)
            rle = mask_utils.merge(rles)
            canvas = np.maximum(canvas, mask_utils.decode(rle))
        return torch.from_numpy(canvas).unsqueeze(0).float()

    def __getitem__(self, index: int):
        key = self.ids[index]

        if self.is_train:
            file_name, height, width = self.image_meta[key]
        else:
            file_name = key

        img_path = self.images_dir / file_name
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image = self.to_tensor(img)  # (1, H, W), float32 in [0, 1]

        if not self.is_train:
            return image, key

        return image, self._build_mask(key, height, width)
    
if __name__ == "__main__":
    data = FilamentDataset(images_dir=Path("filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\train_images"), json_path=Path("filament-segmentation-2026\\MAGFiLO_1.0_Kaggle_2026\\train\\MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    data_to_plot = 0
    print(len(data))
    print(data[data_to_plot])
    
    img, mask = data[data_to_plot]
    img = img.numpy().squeeze()
    mask = mask.numpy().squeeze()
    cv.imshow("Image", img)
    cv.imshow("Mask", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    overlay = cv.addWeighted(img, 0.5, mask, 0.5, 0)
    cv.imshow("Overlay", overlay)
    cv.waitKey(0)
    cv.destroyAllWindows()
    