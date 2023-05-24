import numpy as np
import tensorflow as tf

import torch
import math

def get_coords(height: int, width: int, batch_size: int = 1):
    # split width into width number of samples (should the end really be width? woudlnt it not be counted?)
    nx = np.linspace(start=0, stop=width, num=width)
    ny = np.linspace(start=0, stop=height, num=height)
    u, v = np.meshgrid(nx, ny)
    coords = np.expand_dims(np.stack((u.flatten(), v.flatten()), axis=-1), 0)
    coords_batched = np.concatenate([coords for _ in range(batch_size)], axis=0)
    return coords_batched


def prepare_query_input(output_disp_height: int, output_disp_width: int, batch_size):
    coords = get_coords(height=output_disp_height, width=output_disp_width, batch_size=batch_size)
    batch_size, n_pts, _ = coords.shape
    coords_tensor = tf.convert_to_tensor(coords)
    return coords_tensor


def prepare_query_input_torch(output_disp_height: int, output_disp_width: int, batch_size, num_samples=200000,
                              num_out=2):
    coords = get_coords(height=output_disp_height, width=output_disp_width, batch_size=batch_size)
    # n_points is width *  height
    batch_size, n_points, _ = coords.shape

    coords = torch.Tensor(coords).float().to(device="cpu") # shape = (batch, n_points, 2)
    coords_changed = torch.reshape(coords, (batch_size, -1, 2))

    # 0th output is are the
    output = torch.zeros(num_out, math.ceil(output_disp_width * output_disp_height / num_samples), num_samples)

    split = torch.split(
        coords.reshape(batch_size, -1, 2), int(num_samples / batch_size), dim=1
    )
    with torch.no_grad():
        for i, p_split in enumerate(split):
            points = torch.transpose(p_split, 1, 2)
            print(points)
            # net.query(points.to(device=cuda))
            # preds = net.get_disparity()
            # confidence = net.get_confidence()
            # output1 = output[0, i, : p_split.shape[1]] = preds.to(device=cuda)
            # output2 = output[1, i, : p_split.shape[1]] = confidence.to(device=cuda)
    # res = []
    # for i in range(num_out):
    #     res.append(output[i].view(1, -1)[:, :n_pts].reshape(-1, height, width))
    # return res
    return output


if __name__ == '__main__':
    # prepare_query_input(output_disp_height=10, output_disp_width=20, batch_size=8)
    prepare_query_input_torch(output_disp_height=10, output_disp_width=20, batch_size=8)
