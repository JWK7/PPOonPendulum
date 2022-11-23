import time
import torch
from torch.optim import Adam
from torch.distributions.normal import Normal

from Modules import NormalModule


# standard torch flow
def normal_forward_and_backpropagation_example():
    for multi_input in [False, True]:
        # initialize the model
        layer = NormalModule(2, 1)
        # pass model parameters to optimizer
        optim = Adam(layer.parameters(), lr=0.1)

        if not multi_input:
            # run forward pass on single input (e.g when sampling from environment)
            out_mean, out_variance = layer(torch.randn(2))
        else:
            # run multiple parallel forward passes
            out_mean, out_variance = layer(torch.randn(1000, 2))

        # forward pass cont
        out_action_distribution = Normal(out_mean, out_variance)
        out_sampled             = out_action_distribution.sample()

        # backpropagation
        logprob = out_action_distribution.log_prob(out_sampled)
        grad = logprob * torch.randn(1)  # dummy reward

        optim.zero_grad()
        grad.mean().backward()
        optim.step()
        print("updated layer.log_std", layer.log_std.grad)
        print()


# use vectorized torch operations when you want to compute multiple outputs at once
# torch will automatically parallelize the computation
def vectorized_vs_nonvectorized():
    for vectorized in [False, True]:
        layer = NormalModule(2, 1)

        if not vectorized:
            t_start = time.time()
            [layer(torch.randn(2)) for x in range(10000)]
            t_end = time.time()
            print("not vectorized", t_end - t_start)
        else:
            t_start = time.time()
            layer(torch.randn(10000, 2))
            t_end = time.time()
            print("vectorized", t_end - t_start)


if __name__ == "__main__":
    normal_forward_and_backpropagation_example()
    vectorized_vs_nonvectorized()