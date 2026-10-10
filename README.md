# ResNet from scratch

The code from the video **Why Deeper Networks Got Worse, and the One Plus Sign That Fixed It**. It rebuilds the idea of the ResNet paper (He, Zhang, Ren, Sun, 2015) on a laptop, in plain Python and PyTorch: plain networks get worse when they get deeper, and one plus sign in every block fixes it.

About 430 lines. No download: the pictures are the 1,797 handwritten digits that come with scikit-learn.

## Run it

Python 3.10 or newer, PyTorch and scikit-learn (`pip install torch scikit-learn`). No GPU is needed.

```bash
sh run_all.sh
```

or one step at a time:

```bash
python3 step1_digits.py
```

Steps 1, 2 and 4 train the networks and write them to `files/`. The later steps and the exercise need those files. The full run takes about 10 minutes on a laptop with 8 processes. `demo-output.txt` holds the output of one full run.

## The files

| File | What it holds |
| --- | --- |
| `nets.py` | the pictures, the block with and without the plus sign, the network, training, the sweep over depths |
| `step1` … `step6` | one measurement for each part of the video |
| `exercise.py` | the exercise from the end of the video |
| `tests.py` | 11 checks of the data, the block and the network |

## The plus sign

```python
def forward(self, x):
    if self.skip:
        return torch.relu(x + self.branch(x))
    return torch.relu(self.branch(x))
```

A block is two layers. Without the skip it replaces its input. With the skip it keeps its input and adds a change. The last layer of every skip block starts at zero, so a new network of any depth gives the same answers as its first and last layer alone.

## The steps and what they measured

Every network is 64 numbers wide and trains for 100 passes over the pictures. For each depth, four step sizes are tried with three random starts each; the table shows the best step size, averaged over the three starts.

| Step | What it does | Measured |
| --- | --- | --- |
| 1 `step1_digits.py` | 1,437 training pictures, 360 test pictures. A network with 4 layers and 13,130 numbers. | 100.0 % of the training pictures, 98.2 % of the test pictures. |
| 2 `step2_deeper.py` | The same network with 8, 16, 32, 64 layers. No skips. | Training pictures: 100, 100, 100, 100 % up to 32 layers, then 42.5 % at 64 layers (three starts: 9.5, 77.9, 40.1 %). Test: 98.2, 97.4, 96.9, 95.9, 42.6 %. With four times the training, 64 layers reach 53.1 % (19.7, 99.7, 40.1 %). |
| 3 `step3_do_nothing.py` | Builds a 64-layer network by hand: the trained 8-layer network plus 56 layers that pass their input through. Then trains plain layers on one job: give back your input. | The network built by hand gives the same answer as the 8-layer one on every picture. Error left when plain layers learn "change nothing": 7.0 % for 2 layers, 13.3 % for 8, 15.9 % for 32, 36.2 % for 62. With skips: 0 % before any training. |
| 4 `step4_plus.py` | The same race with the plus sign: 4 to 128 layers. | 100 % of the training pictures at every depth, in every start. Test: 97.6, 98.0, 97.9, 97.4, 97.6, 96.9 %. |
| 5 `step5_signal.py` | Sends pairs of different digits through a new 64-layer network and measures how alike they are after every block. Measures the error signal at every block. | Plain: 0.30 alike after the first layer, 0.95 after layer 17, 0.994 after layer 63. Skip: 0.30 at every layer. The first block of the plain network still gets 56 % of the error signal of the last block. |
| 6 `step6_delete.py` | Deletes blocks from a trained network, with no new training. | Skip network, 64 layers: 97.8 % with all blocks, 97.8 % on average with one block deleted (worst 97.2 %), 97.1 % with 8 of 31 deleted. Plain network, 32 layers: 94.2 % with all blocks, 18.0 % on average with one deleted. One skip block changes its input by 16.6 % on average. |

## The exercise

The trained skip network has 64 layers, so 31 blocks. Delete 16 of them at random, with no new training. What score is left on the test pictures?

```bash
python3 exercise.py
```

<details>
<summary>Answer</summary>

95.4 % on average over 200 draws (worst 91.4 %, best 97.8 %), down from 97.8 % with all blocks.

It matters which blocks go. Without the first 16 blocks the score is 89.4 %. Without the last 16 it is 96.9 %. With all 31 deleted, the first and the last layer alone still score 78.1 %.

</details>

## Limits

- The task is tiny. Deeper networks do not score higher on the test here. The code shows only that depth can be trained with the plus sign and not without it.
- The layers are fully connected. The paper uses convolutional layers, which are made for images.
- No batch normalisation. The plain networks of the paper used it and still got worse with depth.
- Starting the last layer of each block at zero is from later work (Goyal et al. 2017; Fixup, Zhang et al. 2019), not from the 2015 paper.
- Your numbers can differ a little between machines.

## Sources

- He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition", arXiv 1512.03385 (2015)
- Srivastava, Greff, Schmidhuber, "Highway Networks", arXiv 1505.00387 (2015)
- Veit, Wilber, Belongie, "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", arXiv 1605.06431 (2016)
- Vaswani et al., "Attention Is All You Need", arXiv 1706.03762 (2017)
- The digits: `sklearn.datasets.load_digits`, from the UCI "Optical Recognition of Handwritten Digits" data (Alpaydin, Kaynak 1998)

## License

MIT, see `LICENSE`.
