import numpy as np
import cv2


def to_numpy(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load image.")
    else:
        return np.array(image)[:-32, :]

reference_matrixes = np.array([to_numpy(f'all_data/f{str(x).zfill(4)}.png') for x in range(1,2001)])
subject_matrixes = np.array([to_numpy(f'all_data/s{str(x).zfill(4)}.png') for x in range(1,2001)])

np.save('reference_matrixes.npy', reference_matrixes)
print(reference_matrixes.dtype)
np.save('subject_matrixes.npy', subject_matrixes)
print(subject_matrixes.dtype)