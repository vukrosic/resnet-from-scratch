"""Step 6. Look inside the trained networks: how big is the change one block makes, and what if we delete a block?"""
import random

import torch

from nets import digits, load, score, save

if __name__ == '__main__':
    xtr, ytr, xte, yte = digits()
    skip, plain = load('skip', 64, True), load('plain', 32, False)
    res = {}
    for name, net in (('skip', skip), ('plain', plain)):
        full = score(net, xte, yte)
        each = [score(net, xte, yte, without=(i,)) for i in range(len(net.blocks))]
        res[name] = {'full': full, 'each': each}
        print(f'{name} network, {2 + 2 * len(net.blocks)} layers, all blocks: {100 * full:.1f} % of the test pictures')
        print(f'  one block deleted: average {100 * sum(each) / len(each):.1f} %, worst {100 * min(each):.1f} %, best {100 * max(each):.1f} %')
    with torch.no_grad():
        h = torch.relu(skip.first(xte))
        change = []
        for block in skip.blocks:
            change.append((block.branch(h).norm() / h.norm()).item())
            h = block(h)
    res['change'] = change
    print(f'size of the change one block adds, as a share of its input: average {100 * sum(change) / len(change):.1f} %, '
          f'largest {100 * max(change):.1f} %, smallest {100 * min(change):.1f} %')
    rng = random.Random(0)
    many = {}
    print('skip network, blocks deleted at random (average of 50 draws):')
    for k in (1, 2, 4, 8, 12, 16, 20, 24, 28, 31):
        accs = [score(skip, xte, yte, without=tuple(rng.sample(range(31), k))) for _ in range(50)]
        many[k] = sum(accs) / len(accs)
        print(f'  {k:>2} of 31 deleted: {100 * many[k]:.1f} %')
    res['many'] = many
    save('step6', res)
