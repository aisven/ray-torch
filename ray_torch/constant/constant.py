import torch


def create_constant_tensor(v):
    return torch.tensor(v, dtype=torch.float, requires_grad=False)


def create_constant_tensor_on_device(v):
    return torch.tensor(v, dtype=torch.float, requires_grad=False, device=device)


# set log levels
log_level_debug = False
log_level_info = True


# check if CUDA is available
if torch.cuda.is_available():
    device_str = "cuda"
    device = torch.device(device_str)
    print("CUDA is available. Using first GPU.")
else:
    device_str = "cpu"
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

print(f"device={device}")


# check for Metal Performance Shaders in case of macOS just out of curiosity
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not " "built with MPS enabled.")
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )


# numerical constants
minus_one_dot_zero = create_constant_tensor_on_device(-1.0)
minus_zero_dot_five = create_constant_tensor_on_device(-0.5)
zero_dot_zero = create_constant_tensor_on_device(0.0)
one_dot_zero = create_constant_tensor_on_device(1.0)
two_dot_zero = create_constant_tensor_on_device(2.0)
four_dot_zero = create_constant_tensor_on_device(4.0)

zero_vector_float = create_constant_tensor_on_device([0.0, 0.0, 0.0])
zero_vector_int = torch.tensor([0.0, 0.0, 0.0], dtype=torch.int, requires_grad=False)

two_55 = create_constant_tensor_on_device(255.0)

one_by_255 = 1.0 / 255.0
one_over_255 = create_constant_tensor_on_device(one_by_255)


# constant matrices
identity_matrix_3_by_3 = torch.eye(3, dtype=torch.float, requires_grad=False, device=device)
