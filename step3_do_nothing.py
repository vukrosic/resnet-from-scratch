"""Step 3. A deep network that is as good as a shallow one must exist. Can training find it?

Part 1 builds it by hand: the trained 8-layer network, then 28 blocks that pass their input through unchanged.
Part 2 asks a plain stack of layers to learn "change nothing" by training.
"""
import torch
import torch.nn as nn

from nets import WIDTH, RATES, Block, Net, digits, load, score, save


def by_hand():
    """A 64-layer plain network with the same answers as the trained 8-layer one."""
    small = load('plain', 8, False)
    big = Net(64, False)
    big.first.load_state_dict(small.first.state_dict())
    big.last.load_state_dict(small.last.state_dict())
    for i, block in enumerate(big.blocks):
        if i < 3:
            block.load_state_dict(small.blocks[i].state_dict())
        else:                                   # a layer that copies: weight 1 from every number to itself
            for layer in (block.a, block.b):
                layer.weight.data = torch.eye(WIDTH)
                layer.bias.data.zero_()
    return small, big


def copy_error(blocks, skip, x, epochs=100):
    """Train a stack of blocks to give back its input. Returns the error left, as a share of the input."""
    best = None
    for rate in RATES:
        torch.manual_seed(0)
        stack = nn.Sequential(*[Block(skip) for _ in range(blocks)])
        opt = torch.optim.SGD(stack.parameters(), lr=rate, momentum=0.9)
        g = torch.Generator().manual_seed(0)
        start = (((stack(x) - x) ** 2).sum() / (x ** 2).sum()).item()
        for _ in range(epochs):
            order = torch.randperm(len(x), generator=g)
            for i in range(0, len(x), 64):
                xb = x[order[i:i + 64]]
                loss = ((stack(xb) - xb) ** 2).mean()
                if not torch.isfinite(loss):
                    break
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            err = (((stack(x) - x) ** 2).sum() / (x ** 2).sum()).item()
        if err == err and (best is None or err < best[1]):
            best = (start, err, rate)
    return best


if __name__ == '__main__':
    torch.set_num_threads(4)
    xtr, ytr, xte, yte = digits()
    small, big = by_hand()
    a, b = score(small, xtr, ytr), score(big, xtr, ytr)
    same = bool((small(xtr).argmax(1) == big(xtr).argmax(1)).all())
    print(f'trained 8-layer network: {100 * a:.1f} % of the training pictures')
    print(f'64 layers built by hand: {100 * b:.1f} %   same answer on every picture: {same}')
    print(f'on the test pictures: {100 * score(small, xte, yte):.1f} % and {100 * score(big, xte, yte):.1f} %')
    print()
    with torch.no_grad():
        h = torch.relu(small.first(xtr))        # what the blocks of the network get to see
    rows = {}
    print('learn to change nothing: error left after training, as a share of the input')
    for blocks in (1, 4, 16, 31):
        s0, err, rate = copy_error(blocks, False, h)
        r0, rerr, _ = copy_error(blocks, True, h, epochs=0)
        rows[2 * blocks] = {'start': s0, 'plain': err, 'skip_start': r0}
        print(f'{2 * blocks:>3} plain layers: start {100 * s0:6.1f} %, after training {100 * err:6.2f} %   with skips, before any training: {100 * r0:.2f} %')
    save('step3', {'small': a, 'hand': b, 'same': same, 'copy': rows})
