import cv2 as cv
import numpy as np

from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector


def process(detector, img, output_size, output_path):
    _, facial5points = detector.detect_faces(img)

    if len(facial5points) < 1:
        # if there is no facial5 point features then the image won't be created
        return False
    facial5points = np.reshape(facial5points[0], (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square
    )

    # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
    dst_img = warp_and_crop_face(
        img, facial5points, reference_pts=reference_5pts, crop_size=output_size
    )
    cv.imwrite(
        output_path,
        dst_img,
    )
    # img = cv.resize(raw, (224, 224))
    # cv.imwrite('images/{}_img.jpg'.format(i), img)
    return True


if __name__ == "__main__":
    detector = MtcnnDetector()

    # for i in range(10):
    filename = "images/1_0_0_20161219161028662.jpg"
    print("Loading image {}".format(filename))
    raw = cv.imread(filename)
    process(raw, output_size=(224, 224))
    process(raw, output_size=(112, 112))
