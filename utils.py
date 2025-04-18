import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('Agg')  # oppure 'Qt5Agg', 'Qt4Agg', a seconda di ciò che hai installato


def create_video(n, frames=100, interval=1000 / 30):
    X = np.random.binomial(1, 0.3, size=(n, n))

    fig, ax = plt.subplots()
    im = ax.imshow(X, cmap='gray', interpolation='nearest')

    def animate(t):
        nonlocal X  # Evita variabili globali
        X = np.roll(X, shift=1, axis=0)  # Shift di una riga in basso
        im.set_array(X)
        return im,

    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)

    # Mostrare il video
    plt.show()

    # Salvataggio del video
    # anim.save("output.mp4", fps=30, extra_args=['-vcodec', 'libx264'])

    return anim


def stateNameToCoords(name):
    if not isinstance(name, str) or not name.startswith('x'):
        return None
    parts = name[1:].split('y')
    if len(parts) != 2:
        return None
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError:
        return None