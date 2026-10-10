"""Step 1. The task and a small network: 4 layers, 1,437 handwritten digits."""
from nets import digits, sweep, table, save, Net

if __name__ == '__main__':
    xtr, ytr, xte, yte = digits()
    print('pictures to learn from:', len(xtr), '  test pictures:', len(xte), '  pixels in a picture:', xtr.shape[1])
    n = sum(p.numel() for p in Net(4, False).parameters())
    print('numbers in the 4-layer network:', f'{n:,}')
    out = sweep([4], skip=False, name='small')
    table(out)
    wrong = round((1 - out[4]['tests'][0]) * len(xte))
    print('first run: wrong on', wrong, 'of', len(xte), 'test pictures')
    save('step1', {'train': len(xtr), 'test': len(xte), 'numbers': n, 'acc_train': out[4]['train'], 'acc_test': out[4]['test'],
                   'wrong': wrong, 'curve': out[4]['curve']})
