from typing import Tuple
import torch


def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def clamp_weights(model: torch.nn.Module, range: Tuple[float, float] = (-0.01, 0.01)):
    assert range[0] < range[1]
    for param in model.parameters():
        param.data.clamp_(range[0], range[1])


def gradient_penalty(netD: torch.nn.Module, real_data: torch.Tensor, fake_data: torch.Tensor, device: torch.device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
