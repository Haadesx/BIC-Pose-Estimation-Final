# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 19:56
# @Author  : Yaozu Ye, Hao Shi, Jiaan Chen

import torch


class RasEventCloud:
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.channels = input_size[0]
        self.height = input_size[1]
        self.width = input_size[2]
        self.e_cloud_list = list()
        self.cloud_channel_grid = torch.ones(input_size, dtype=torch.float, requires_grad=False)
        for i in range(self.channels):
            self.cloud_channel_grid[i] *= i

        y_cord = torch.tensor([i for i in range(self.height)])
        x_cord = torch.tensor([i for i in range(self.width)])
        y, x = torch.meshgrid(y_cord, x_cord)
        y = y.unsqueeze(0)
        x = x.unsqueeze(0)
        self.y_grid = y.expand(self.channels, -1, -1)
        self.x_grid = x.expand(self.channels, -1, -1)

    def convert(self, events):
        """
        return data: channels, x, y, t_avg, p_acc, e_count
        """
        events = torch.from_numpy(events)
        C, H, W = self.channels, self.height, self.width
        cloud_polar_grid = torch.zeros((C, H, W), dtype=torch.float, requires_grad=False)
        cloud_time_acc_grid = torch.zeros((C, H, W), dtype=torch.float, requires_grad=False)
        cloud_event_count_grid = torch.zeros((C, H, W), dtype=torch.float, requires_grad=False)

        events[:, 2] = events[:, 2] - events[:, 2].min()
        events[:, 2] = (events[:, 2] / (events[:, 2].max())) / 1.001
        events[:, 2] = events[:, 2] * self.channels

        with torch.no_grad():
            t = events[:, 2]
            p = 2 * events[:, 3] - 1
            t0 = events[:, 2].int()
            
            # Original logic used -1 shift. Preserving it but filtering the result.
            x = events[:, 0].int() - 1 
            y = events[:, 1].int() - 1

            # Bounds checking
            mask = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (t0 >= 0) & (t0 < C)
            
            # Filter valid events
            t = t[mask]
            p = p[mask]
            t0 = t0[mask]
            x = x[mask]
            y = y[mask]

            if x.numel() > 0: # Ensure we have data
                index = H * W * t0.long() + W * y.long() + x.long()
                cloud_polar_grid.put_(index, p.float(), accumulate=True)
                cloud_time_acc_grid.put_(index, (t - t0).float(), accumulate=True)
                cloud_event_count_grid.put_(index, torch.ones_like(x, dtype=torch.float), accumulate=True)

            valid_mask = torch.ne(cloud_event_count_grid, 0)
            data_list = list()
            data_list.append(self.cloud_channel_grid[valid_mask].reshape(-1, 1))
            data_list.append(self.x_grid[valid_mask].reshape(-1, 1))
            data_list.append(self.y_grid[valid_mask].reshape(-1, 1))
            data_list.append(cloud_time_acc_grid[valid_mask].reshape(-1, 1))
            data_list.append(cloud_polar_grid[valid_mask].reshape(-1, 1))
            data_list.append(cloud_event_count_grid[valid_mask].reshape(-1, 1))

            data = torch.cat(data_list, 1)
            
            # Avoid division by zero
            if data.shape[0] > 0:
                data[:, 3] /= data[:, 5]
                data[:, 3] = data[:, 3] / C + data[:, 0] / C
            else:
                 # If everything filtered out, return empty with correct shape
                return torch.empty((0, 6), dtype=torch.float)

            return data
