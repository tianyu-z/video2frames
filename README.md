# video2frames


## 1. Determine the bounding box with several samples

If one knows the bounding box of the selected regions, one may skip this step. If there are no bounding box needs to be selected, please don't specify it. It will be fine with default value ``None``. 

``python .\get_frames_from_video.py --i_path "\path\to\videos" --o_path "path\to\frames"  --sample_interval_or_num_samples 10 --is_n_sample 1 ``


This will sample 10 frames from the video uniformly. One may check out the output folders and decide ``--x_min``, ``--x_max``, ``--y_min``, ``--x_max``. (One may refer to the ``sample_pic.png`` as a reference of axis to determine the bounding box.


## 2. Output all the interest frames according to the bounding box

``python .\get_frames_from_video.py --i_path "\path\to\videos" --o_path "path\to\frames"  --sample_interval_or_num_samples 20 --is_n_sample 0 --x_min a --x_max b --y_min c --y_max d --sensitivity e``


This will sample every 20 seconds uniformly from the video, cut it with the bounding box ``(a, b, c, d)`` and automatically filter out very similar frames by ``sensitivity e``. This is how ``sensitivity`` works: calculate the percentage difference between 2 continuous samples. If it is smaller than ``sensitivity e``, it will be filtered out. Empirically speaking ``sensitivity`` from ``0.01`` to ``0.12`` is popular for slides. If one doesn't specify the ``sensitivity``, it will set the most frequent difference percentage as the ``sensitivity``.
