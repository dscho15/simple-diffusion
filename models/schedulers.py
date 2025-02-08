import torch
from functools import partial
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import tan, pi

# Linear Schedule


def linear_schedulee(step):
    return (1 - 0) * step


# Sigmoid Schedule
# shift=64 is ref size https://arxiv.org/pdf/2301.11093 eq (5)


def log(x):
    return torch.log(x)


def log_snr(step, d, shift):
    ratio = torch.tensor(shift / d, dtype=torch.float32)
    return -2 * log(tan(pi * step / 2)) + 2 * log(ratio)


def sigmoid_schedule(step, img_size=128, shift=64):
    return torch.sigmoid(-log_snr(step, img_size, shift))


def visualize_schedule(schedule, total_steps=1000):
    t = torch.linspace(0, 1, total_steps)
    schedule_values = schedule(t)
    plt.plot(t, schedule_values)
    plt.savefig(schedule.__name__ + ".png")
    plt.clf()


def interpolate_schedules(schedule1, schedule2, t):
    def interpolated_schedule(t):
        return t * schedule1(t) + (1 - t) * schedule2(t)

    return interpolated_schedule


# visualize_schedule(linear_schedulee, 1000)
# visualize_schedule(sigmoid_schedule, 1000)
# visualize_schedule(interpolate_schedules(partial(sigmoid_schedule, img_size=512, shift=32), partial(sigmoid_schedule, img_size=512, shift=64), 0.5), 1000)
