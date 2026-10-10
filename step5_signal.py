"""Step 5. Why the plain network fails: what happens to two different pictures on the way through 64 new layers,
and how much of the error signal comes back to the first layers."""
import torch
import torch.nn as nn

from nets import Net, digits, load, save


def alike(net, x, pairs):
    """After the first layer and after every block: how alike are the hidden numbers of two different digits?
    1 means the two pictures have become the same list of numbers (up to its length)."""
    out = []
    with torch.no_grad():
        h = torch.relu(net.first(x))
        out.append(nn.functional.cosine_similarity(h[pairs[0]], h[pairs[1]]).mean().item())
        for block in net.blocks:
            h = block(h)
            out.append(nn.functional.cosine_similarity(h[pairs[0]], h[pairs[1]]).mean().item())
    return out


def signal(net, x, y):
    """Size of the error signal that arrives at the input of every block, for one pass over all pictures."""
    hs = []
    h = torch.relu(net.first(x))
    for block in net.blocks:
        h.retain_grad()
        hs.append(h)
        h = block(h)
    h.retain_grad()
    hs.append(h)
    nn.CrossEntropyLoss()(net.last(h), y).backward()
    return [t.grad.norm().item() for t in hs]


if __name__ == '__main__':
    xtr, ytr, _, _ = digits()
    g = torch.Generator().manual_seed(0)
    i, j = torch.randperm(len(xtr), generator=g)[:400], torch.randperm(len(xtr), generator=g)[:400]
    keep = ytr[i] != ytr[j]
    pairs = (i[keep], j[keep])
    res = {}
    for name, skip in (('plain', False), ('skip', True)):
        sims, sigs = [], []
        for seed in range(5):
            torch.manual_seed(seed)
            net = Net(64, skip)
            sims.append(alike(net, xtr, pairs))
            sigs.append(signal(net, xtr, ytr))
        sim = [sum(s[k] for s in sims) / 5 for k in range(len(sims[0]))]
        sig = [sum(s[k] for s in sigs) / 5 for k in range(len(sigs[0]))]
        res[name] = {'alike': sim, 'signal': sig}
        print(name, 'network, 64 layers, new (5 networks averaged)')
        print('  two different digits, how alike: after layer 1 %.3f, layer 17 %.3f, layer 33 %.3f, layer 63 %.4f'
              % (sim[0], sim[8], sim[16], sim[-1]))
        print('  error signal: at the last block %.4f, at the first block %.6f, first / last = %.2f' % (sig[-1], sig[0], sig[0] / sig[-1]))
    for name, skip in (('plain', False), ('skip', True)):
        sim = alike(load(name, 64, skip), xtr, pairs)
        res[name]['trained'] = sim
        print(name, 'network, 64 layers, trained: after layer 1 %.3f, layer 33 %.3f, layer 63 %.3f' % (sim[0], sim[16], sim[-1]))
    save('step5', res)
