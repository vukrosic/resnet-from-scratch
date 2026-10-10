"""Checks of the data and the networks. Run: python3 tests.py"""
import torch

from nets import WIDTH, Block, Net, digits, score
from step3_do_nothing import by_hand

xtr, ytr, xte, yte = digits()
checks = []


def check(name, ok):
    checks.append(ok)
    print('ok  ' if ok else 'FAIL', name)


check('1,437 training pictures and 360 test pictures', len(xtr) == 1437 and len(xte) == 360)
check('a picture is 64 pixels', xtr.shape[1] == 64)
check('all ten digits are in the test set', len(set(yte.tolist())) == 10)
check('a 64-layer network has 31 blocks', len(Net(64, False).blocks) == 31)
check('64 layers hold 262,730 numbers', sum(p.numel() for p in Net(64, False).parameters()) == 262730)
check('plain and skip networks hold the same count of numbers',
      sum(p.numel() for p in Net(32, False).parameters()) == sum(p.numel() for p in Net(32, True).parameters()))
torch.manual_seed(0)
h = torch.rand(5, WIDTH)
check('a new block with a skip changes nothing', torch.equal(Block(True)(h), h))
check('a new plain block changes its input', not torch.allclose(Block(False)(h), h))
net = Net(16, True)
check('deleting a block that changes nothing changes no answer', torch.equal(net(xte), net(xte, without=(3,))))
check('a new network guesses about 1 in 10', 0.0 <= score(Net(8, False), xte, yte) < 0.4)
try:
    small, big = by_hand()
    check('the 64-layer network built by hand answers like the 8-layer one', bool((small(xte).argmax(1) == big(xte).argmax(1)).all()))
except FileNotFoundError:
    print('skip the by-hand check: run step2_deeper.py first')
print(sum(checks), 'of', len(checks), 'checks passed')
