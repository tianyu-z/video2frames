import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--i_path", type=str, default="/", help="the input video path")
parser.add_argument("--o_path", type=str, default="/", help="the output pics path")
parser.add_argument(
    "--sample_interval_or_num_samples",
    type=int,
    default=None,
    help="the number_of_interval (seconds) or number of pictures one expects to get (it is the upper bound because there will be a lot of them filtered out later, ~300 is a safe number)",
)
parser.add_argument(
    "--is_n_sample",
    type=int,
    default=0,
    help="whether the sample_interval_or_num_samples is the_number_of_samples or not, boolean, default is False (0)",
)
parser.add_argument(
    "--start",
    type=int,
    default=None,
    help="the start second of the video, int, default is None, if None, it will be 0",
)
parser.add_argument(
    "--end",
    type=int,
    default=None,
    help="the end second of the video, int, default is None, if None, it will be the last second of the video",
)
parser.add_argument(
    "--x_min",
    type=int,
    default=None,
    help="the minimum x of the bounding box, int, default is None, if None, it will be set to 0",
)
parser.add_argument(
    "--x_max",
    type=int,
    default=None,
    help="the maximum x of the bounding box, int, default is None, if None, it will be set to the width of the frame",
)
parser.add_argument(
    "--y_min",
    type=int,
    default=None,
    help="the minimum y of the bounding box, int, default is None, if None, it will be set to 0",
)
parser.add_argument(
    "--y_max",
    type=int,
    default=None,
    help="the maximum y of the bounding box, int, default is None, if None, it will be set to the height of the frame",
)
parser.add_argument(
    "--sensitivity",
    type=float,
    default=None,
    help="the sensitivity filter repeat slides, float, default is None. If it's None, it will automatically get a safe sensitivity",
)

args = parser.parse_args()


def compare(a, b):
    return np.abs(np.sum(a - b)) / np.sum(b)


def get_frames(
    i_path,
    o_path,
    sample_interval_or_num_samples,
    is_n_sample=False,
    start=None,
    end=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    sensitivity=None,
):
    """
    This function is used to get the slides from the video.
    @param i_path: the input video path
    @param o_path: the output pics path
    @param sample_interval_or_num_samples: the number_of_interval (seconds) or number of pictures one expects to get
                                           (it is the upper bound because there will be a lot of them filtered out later, ~300 is a safe number)
    @param is_n_sample: whether the sample_interval_or_num_samples is the_number_of_samples or not, boolean, default is False
    @param start: the start second of the video, int, default is None, if None, it will be 0
    @param end: the end second of the video, int, default is None, if None, it will be the last second of the video
    @param x_min: the minimum x of the bounding box, int, default is None, if None, it will be set to 0
    @param x_max: the maximum x of the bounding box, int, default is None, if None, it will be set to the width of the frame
    @param y_min: the minimum y of the bounding box, int, default is None, if None, it will be set to 0
    @param y_max: the maximum y of the bounding box, int, default is None, if None, it will be set to the height of the frame
    @param sensitivity: the sensitivity filter repeat slides, float, default is None. If it's None, it will automatically get a safe sensitivity but it is usually too low, so you probably need to filter pics manually later
    """
    vidcap = cv2.VideoCapture(i_path)
    success, image = vidcap.read()

    if x_min is None:
        x_min = 0
    if y_min is None:
        y_min = 0
    if x_max is None:
        x_max = image.shape[0]
    if y_max is None:
        y_max = image.shape[1]

    frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if start is None:
        start = 0
    if end is None:
        end = frames / fps
    images = []

    if is_n_sample:
        sample_interval = end // sample_interval_or_num_samples
    else:
        sample_interval = sample_interval_or_num_samples

    for i in range(int(start * fps), int(end * fps) + 1, int(sample_interval * fps)):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        images.append(image[x_min:x_max, y_min:y_max, :])
        # cv2.imwrite(os.path.join(o_path, "frame_{}.jpg".format(i)), image[x_min:x_max,y_min:y_max,:])     # save frame as JPEG file

    if sensitivity is None:
        diff = [compare(images[i], images[i + 1]) for i in range(len(images) - 1)]
        c = plt.hist(diff, bins=30)
        sensitivity = c[1][np.argmax(c[0])]
        print("Find sensitivity: ", sensitivity)

    old = images[0]
    new = images[1]
    filtered_images = []
    for i in range(len(images)):
        new = images[i]
        if np.abs(np.sum(new - old)) > (sensitivity + 0.001) * np.sum(old):
            filtered_images.append(new)
        old = new
    # save
    for i in range(len(filtered_images)):
        cv2.imwrite(os.path.join(o_path, "frame_{}.jpg".format(i)), filtered_images[i])
    return images, filtered_images


dict_vars = vars(args)
print(dict_vars)
get_frames(**dict_vars)
