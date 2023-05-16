"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(lambda: 0)
        self.weight_decay = weight_decay
        self.t = 0

    def step(self):
        u = self.u
        for p in self.params:
            if p.grad:
                grad = p.grad.numpy().astype(p.data.dtype)
                # Weight decay, penalize according to weight values
                if self.weight_decay > 0:
                    grad += self.weight_decay * p.numpy()
                u[p] = u[p] * self.momentum + (1 - self.momentum) * grad
                p.data -= (self.lr * u[p])

        self.t += 1


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(lambda: 0)
        self.v = defaultdict(lambda: 0)

    def step(self):
        self.t += 1
        m = self.m
        v = self.v

        for p in self.params:
            if p.grad:
                grad = p.grad.data
                if self.weight_decay > 0:
                    grad += self.weight_decay * p.data
                m[p] = m[p] * self.beta1 + (1 - self.beta1) * grad
                v[p] = v[p] * self.beta2 + (1 - self.beta2) * (grad**2)
                m_corr = m[p] / (1 - self.beta1**self.t) # bias correction
                v_corr = v[p] / (1 - self.beta2**self.t)
                delta = ndl.Tensor(self.lr * (m_corr/(v_corr**0.5 + self.eps)), dtype=p.data.dtype)
                p.data -= delta