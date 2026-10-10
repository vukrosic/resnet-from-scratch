"""The exercise from the end of the video.

The trained network with skips has 31 blocks. Delete 16 of them at random, and train nothing again.
How many of the 360 test pictures does it still get right? Guess first, then run this file.
"""
import random

from nets import digits, load, score

if __name__ == '__main__':
    xtr, ytr, xte, yte = digits()
    net = load('skip', 64, True)
    print(f'all 31 blocks: {100 * score(net, xte, yte):.1f} % of the test pictures')
    rng = random.Random(1)
    for k in (16, 31):
        accs = [score(net, xte, yte, without=tuple(rng.sample(range(31), k))) for _ in range(200)]
        print(f'{k} of 31 blocks deleted at random, 200 draws: average {100 * sum(accs) / len(accs):.1f} %, '
              f'worst {100 * min(accs):.1f} %, best {100 * max(accs):.1f} %')
    print('the second line is the first layer and the last layer alone')
    front = score(net, xte, yte, without=tuple(range(16)))
    back = score(net, xte, yte, without=tuple(range(15, 31)))
    print(f'the first 16 blocks deleted: {100 * front:.1f} %   the last 16 blocks deleted: {100 * back:.1f} %')
