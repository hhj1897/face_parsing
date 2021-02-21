import numpy as np

random_state = np.random.RandomState(seed=1234)


def label_colormap(n_label=11):
    """Label colormap.
    Parameters
    ----------
    n_labels: int
        Number of labels (default: 11).
    value: float or int
        Value scale or value of label color in HSV space.
    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.
    """
    if n_label == 11:  # helen, ibugmask
        cmap = np.array(
            [
                (0, 0, 0),
                (255, 255, 0),
                (139, 76, 57),
                (139, 54, 38),
                (0, 205, 0),
                (0, 138, 0),
                (154, 50, 205),
                (72, 118, 255),
                (255, 165, 0),
                (0, 0, 139),
                (255, 0, 0),
            ],
            dtype=np.uint8,
        )
    elif n_label == 19:  # celebamask-hq
        cmap = np.array(
            [
                (0, 0, 0),
                (204, 0, 0),
                (76, 153, 0),
                (204, 204, 0),
                (51, 51, 255),
                (204, 0, 204),
                (0, 255, 255),
                (255, 204, 204),
                (102, 51, 0),
                (255, 0, 0),
                (102, 204, 0),
                (255, 255, 0),
                (0, 0, 153),
                (0, 0, 204),
                (255, 51, 153),
                (0, 204, 204),
                (0, 51, 0),
                (255, 153, 51),
                (0, 204, 0),
            ],
            dtype=np.uint8,
        )
    else:

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = np.zeros((n_label, 3), dtype=np.uint8)
        for i in range(n_label):
            id = i
            r, g, b = 0, 0, 0
            for j in range(8):
                r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
                g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
                b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

    return cmap
