import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def animate_woa_contour(trajectory, fitness_func, bounds, interval=500):
    fig, ax = plt.subplots()

    # 构建背景等高线
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = np.array([fitness_func([xx, yy]) for xx, yy in zip(X_grid.ravel(), Y_grid.ravel())]).reshape(X_grid.shape)
    contour = ax.contourf(X_grid, Y_grid, Z, cmap='Blues')
    plt.colorbar(contour)

    scat = ax.scatter([], [], color='black')

    def update(frame):
        positions = trajectory[frame]
        scat.set_offsets(positions[:, :2])  # 只画前两个维度
        ax.set_title(f"Iteration {frame}")
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    ani.save("woa_animation.gif", writer="pillow")
    plt.show()
