import torch

from ray_torch.utility.utility import create_tensor_on_device


minus_one_dot_zero = create_tensor_on_device(-1.0)
minus_zero_dot_five = create_tensor_on_device(-0.5)
zero_dot_zero = create_tensor_on_device(0.0)
one_dot_zero = create_tensor_on_device(1.0)
two_dot_zero = create_tensor_on_device(2.0)
four_dot_zero = create_tensor_on_device(4.0)

zero_vector_float = create_tensor_on_device([0.0, 0.0, 0.0])
zero_vector_int = torch.tensor([0.0, 0.0, 0.0], dtype=torch.int, requires_grad=False)

two_55 = create_tensor_on_device(255.0)

one_by_255 = 1.0 / 255.0
one_over_255 = create_tensor_on_device(one_by_255)
