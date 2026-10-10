"""The networks of the video: a plain stack of layers, and the same stack with skip connections."""
import json
import os
from multiprocessing import Pool

import torch
import torch.nn as nn
from sklearn.datasets import load_digits

WIDTH = 64                          # numbers in every hidden layer
RATES = (0.003, 0.01, 0.03, 0.1)    # step sizes we try for every network
SEEDS = (0, 1, 2)                   # every network is trained three times
FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')


def digits():
    """1,797 handwritten digits, 8 by 8 pixels. 1,437 to learn from, 360 kept apart as a test."""
    d = load_digits()
    x = torch.tensor(d.data, dtype=torch.float32) / 16.0
    y = torch.tensor(d.target)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    tr, te = order[:1437], order[1437:]
    x = x - x[tr].mean(0)
    return x[tr], y[tr], x[te], y[te]


class Block(nn.Module):
    """Two layers. With skip=True the input is added back to what the two layers made."""

    def __init__(self, skip):
        super().__init__()
        self.skip = skip
        self.a = nn.Linear(WIDTH, WIDTH)
        self.b = nn.Linear(WIDTH, WIDTH)
        nn.init.kaiming_normal_(self.a.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.b.weight, nonlinearity='relu')
        nn.init.zeros_(self.a.bias)
        nn.init.zeros_(self.b.bias)
        if skip:
            nn.init.zeros_(self.b.weight)   # the block starts as "change nothing"

    def branch(self, x):
        return self.b(torch.relu(self.a(x)))

    def forward(self, x):
        if self.skip:
            return torch.relu(x + self.branch(x))   # the plus sign
        return torch.relu(self.branch(x))


class Net(nn.Module):
    """depth counts the layers with weights: one at the start, two in every block, one at the end."""

    def __init__(self, depth, skip):
        super().__init__()
        self.first = nn.Linear(64, WIDTH)
        nn.init.kaiming_normal_(self.first.weight, nonlinearity='relu')
        nn.init.zeros_(self.first.bias)
        self.blocks = nn.ModuleList(Block(skip) for _ in range((depth - 2) // 2))
        self.last = nn.Linear(WIDTH, 10)

    def forward(self, x, without=()):
        x = torch.relu(self.first(x))
        for i, block in enumerate(self.blocks):
            if i not in without:            # "without" deletes blocks, for step 6
                x = block(x)
        return self.last(x)


def score(net, x, y, without=()):
    """Share of the pictures the network names correctly."""
    net.eval()
    with torch.no_grad():
        return (net(x, without).argmax(1) == y).float().mean().item()


def train(net, rate, epochs, seed, x, y):
    """Plain gradient descent with momentum, 64 pictures at a time. Returns the loss after every epoch."""
    opt = torch.optim.SGD(net.parameters(), lr=rate, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 64):
            idx = order[i:i + 64]
            loss = loss_fn(net(x[idx]), y[idx])
            if not torch.isfinite(loss):
                return curve + [float('nan')]
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            curve.append(loss_fn(net(x), y).item())
    return curve


def one(depth, skip, rate, seed, epochs):
    torch.set_num_threads(1)
    xtr, ytr, xte, yte = digits()
    torch.manual_seed(seed)
    net = Net(depth, skip)
    curve = train(net, rate, epochs, seed, xtr, ytr)
    bad = curve[-1] != curve[-1]
    return {'loss': curve[-1], 'curve': curve, 'state': net.state_dict(),
            'train': 0.1 if bad else score(net, xtr, ytr), 'test': 0.1 if bad else score(net, xte, yte)}


def sweep(depths, skip, epochs=100, name=None):
    """Train every depth with every step size, three times each. Keep the step size with the lowest loss."""
    jobs = [(d, skip, r, s, epochs) for d in depths for r in RATES for s in SEEDS]
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        res = pool.starmap(one, jobs)
    out = {}
    for d in depths:
        best = None
        for r in RATES:
            runs = [x for j, x in zip(jobs, res) if j[0] == d and j[2] == r]
            loss = sum(x['loss'] for x in runs) / len(runs)
            if loss == loss and (best is None or loss < best[0]):
                best = (loss, r, runs)
        loss, r, runs = best
        out[d] = {'rate': r, 'loss': loss,
                  'train': sum(x['train'] for x in runs) / len(runs), 'test': sum(x['test'] for x in runs) / len(runs),
                  'trains': [x['train'] for x in runs], 'tests': [x['test'] for x in runs], 'curve': runs[0]['curve']}
        if name:
            os.makedirs(FILES, exist_ok=True)
            torch.save(runs[0]['state'], os.path.join(FILES, f'{name}-{d}.pt'))
    if name:
        save(name, {str(d): {k: v for k, v in o.items()} for d, o in out.items()})
    return out


def load(name, depth, skip):
    net = Net(depth, skip)
    net.load_state_dict(torch.load(os.path.join(FILES, f'{name}-{depth}.pt')))
    return net.eval()


def save(name, data):
    os.makedirs(FILES, exist_ok=True)
    with open(os.path.join(FILES, name + '.json'), 'w') as f:
        json.dump(data, f)


def read(name):
    with open(os.path.join(FILES, name + '.json')) as f:
        return json.load(f)


def table(out):
    print('layers   step size   right on training pictures   right on test pictures')
    for d, o in out.items():
        print(f'{d:>6}   {o["rate"]:>9}   {100 * o["train"]:>25.1f} %   {100 * o["test"]:>20.1f} %')
