import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face



reference = get_reference_facial_points(default_square= True)

print(reference)
# warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(112,112))