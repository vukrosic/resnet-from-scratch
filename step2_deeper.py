"""Step 2. Stack more layers. Plain networks of 4, 8, 16, 32 and 64 layers, same data, same training."""
from nets import sweep, table, save

DEPTHS = [4, 8, 16, 32, 64]

if __name__ == '__main__':
    out = sweep(DEPTHS, skip=False, name='plain')
    table(out)
    print()
    print('Is 64 layers only slow? The same network with four times the training (400 epochs):')
    long = sweep([64], skip=False, epochs=400)
    table(long)
    save('step2', {'long_train': long[64]['train'], 'long_test': long[64]['test'], 'long_trains': long[64]['trains'],
                   'long_rate': long[64]['rate'], 'long_curve': long[64]['curve']})
