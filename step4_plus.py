"""Step 4. The same networks with one change: every block adds its input back to its result."""
from nets import sweep, table, digits, Net, score
import torch

DEPTHS = [4, 8, 16, 32, 64, 128]

if __name__ == '__main__':
    xtr, ytr, xte, yte = digits()
    torch.manual_seed(0)
    new, shallow = Net(64, True), Net(2, True)
    shallow.first.load_state_dict(new.first.state_dict())
    shallow.last.load_state_dict(new.last.state_dict())
    print('a new 64-layer network with skips gives the same answers as its 2 outer layers alone:',
          bool(torch.allclose(new(xtr), shallow(xtr))))
    out = sweep(DEPTHS, skip=True, name='skip')
    table(out)
