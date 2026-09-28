"""Segment nuclei in RGB MIP images using StarDist."""

from pathlib import Path

import numpy as np
import tifffile
from csbdeep.utils import normalize
from skimage.measure import regionprops
from stardist.models import StarDist2D
from tqdm import tqdm


INPUT_DIR = Path("path/to/mip_images")
OUTPUT_DIR = Path("path/to/nuclei_masks")
GROUPS = ["group_1", "group_2"]

NUCLEUS_CHANNEL = 2
PROB_THRESHOLD = 0.479071
NMS_THRESHOLD = 0.3
MIN_NUCLEUS_AREA = 40


def segment_nuclei():
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    for group in GROUPS:
        input_dir = INPUT_DIR / group
        output_dir = OUTPUT_DIR / group / "mask"
        output_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(
            list(input_dir.glob("*.tif"))
            + list(input_dir.glob("*.tiff"))
        )

        for path in tqdm(tif_files, desc=group):
            image = tifffile.imread(path)

            if image.ndim == 2:
                nucleus = image
            elif image.ndim == 3 and image.shape[-1] <= 4:
                nucleus = image[..., NUCLEUS_CHANNEL]
            elif image.ndim == 3:
                nucleus = image[NUCLEUS_CHANNEL]
            else:
                continue

            nucleus = normalize(
                nucleus,
                1,
                99.8,
                axis=(0, 1),
            )

            labels, _ = model.predict_instances(
                nucleus,
                prob_thresh=PROB_THRESHOLD,
                nms_thresh=NMS_THRESHOLD,
            )

            valid_labels = []
            height, width = labels.shape

            for prop in regionprops(labels):
                min_r, min_c, max_r, max_c = prop.bbox

                touches_border = (
                    min_r == 0
                    or min_c == 0
                    or max_r == height
                    or max_c == width
                )

                if not touches_border and prop.area > MIN_NUCLEUS_AREA:
                    valid_labels.append(prop.label)

            mask = np.zeros_like(labels, dtype=np.uint16)

            for label_id in valid_labels:
                mask[labels == label_id] = label_id

            tifffile.imwrite(
                output_dir / f"{path.stem}_mask.tif",
                mask,
            )


if __name__ == "__main__":
    segment_nuclei()
