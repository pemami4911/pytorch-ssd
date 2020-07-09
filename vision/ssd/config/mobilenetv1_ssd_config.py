import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
#   SSDSpec(19, 16, SSDBoxSizes(60, 105), [1.5, 3]),
#    SSDSpec(10, 32, SSDBoxSizes(105, 150), [1.5, 3]),
#    SSDSpec(5, 64, SSDBoxSizes(150, 195), [1.5, 3]),
#    SSDSpec(3, 100, SSDBoxSizes(195, 240), [1.5, 3]),
#    SSDSpec(2, 150, SSDBoxSizes(240, 285), [1.5, 3]),
#    SSDSpec(1, 300, SSDBoxSizes(285, 330), [1.5, 3])
   SSDSpec(19, 16, SSDBoxSizes(30, 52), [1.5, 3]),
    SSDSpec(10, 32, SSDBoxSizes(52, 75), [1.5, 3]),
    SSDSpec(5, 64, SSDBoxSizes(75, 98), [1.5, 3]),
    SSDSpec(3, 100, SSDBoxSizes(98, 120), [1.5, 3]),
    SSDSpec(2, 150, SSDBoxSizes(120, 143), [1.5, 3]),
    SSDSpec(1, 300, SSDBoxSizes(143, 165), [1.5, 3])
]


priors = generate_ssd_priors(specs, image_size)
