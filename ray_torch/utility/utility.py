import torch
import torch.nn.functional as tf

from ray_torch.constant.constant import device
from ray_torch.constant.constant import device_str
from ray_torch.constant.constant import log_level_debug
from ray_torch.constant.constant import log_level_info


def create_tensor(v):
    return torch.tensor(v, dtype=torch.float, requires_grad=False)


def create_tensor_on_device(v):
    return torch.tensor(v, dtype=torch.float, requires_grad=False, device=device)


def is_float_tensor(obj):
    return isinstance(obj, torch.FloatTensor) or isinstance(obj, torch.cuda.FloatTensor)


def is_float_tensor_on_device(obj):
    # obj.device.type is a str
    return is_float_tensor(obj) and (device_str == obj.device.type)


def is_int_tensor(obj):
    return isinstance(obj, torch.IntTensor) or isinstance(obj, torch.cuda.LongTensor)


def is_int_tensor_on_device(obj):
    # obj.device.type is a str
    return is_int_tensor(obj) and (device_str == obj.device.type)


def is_long_tensor(obj):
    return isinstance(obj, torch.LongTensor) or isinstance(obj, torch.cuda.LongTensor)


def is_long_tensor_on_device(obj):
    # obj.device.type is a str
    return is_long_tensor(obj) and (device_str == obj.device.type)


def normalize_vector_custom(v):
    return v / torch.max(torch.norm(v), torch.tensor(1e-12, dtype=torch.float))


def normalize_vector(v):
    return tf.normalize(v, dim=0)


def mean_ignoring_zero(t):
    mask = t != 0.0
    t_mean = (t * mask).sum(dim=0) / mask.sum(dim=0)
    return t_mean


def see(name, value, critical=True):
    if log_level_debug or (critical and log_level_info):
        if is_float_tensor(value):
            if len(value.shape) == 0:
                # scalar
                print(f"{name}={value}")
            elif len(value.shape) == 1:
                if value.shape[0] >= 2:
                    # vector
                    print(f"torch.norm({name})={torch.norm(value)}")
                print(f"{name}.shape={value.shape}")
                print(f"{name}={value}")
            else:
                # matrix or higher-dimensional tensor
                print(f"{name}.shape={value.shape}")
                print(f"{name}=\n{value}")
        else:
            print(f"{name}={value}")


def see_more(name, value, critical=True):
    if log_level_debug or (critical and log_level_info):
        see(name, value, critical)
        if len(value.shape) > 0:
            print(f"{name}.min()={value.min()}")
            print(f"{name}.mean()={value.mean(dtype=torch.float)}")
            print(f"mean_ignoring_zero({name})={mean_ignoring_zero(value)}")
            # print(f"{name}.mode()={value.mode()}")
            print(f"{name}.median()={value.median()}")
            print(f"{name}.max()={value.max()}")
