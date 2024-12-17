import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from typing import List, Callable, Optional, Tuple
import os
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

def plotXyVectorizedHeatmapMovieOnGrid(
    functionHistory: List[Tuple[float, Callable[[float, float], float]]], # Callable should expect to receive X and Y as np.array
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nPartitionPerAxis: int,
    title: Optional[str] = None,
    saveAddr: Optional[str] = None,
    show: bool = False,
    interval: int = 500,
    cmap = 'winter',
    **kwargs
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # I like to position my colorbars this way, but you don't have to
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    N = 100
    x = np.linspace(xmin, xmax, nPartitionPerAxis)
    y = np.linspace(ymin, ymax, nPartitionPerAxis)
    X, Y = np.meshgrid(x, y)

    # This is now a list of arrays rather than a list of artists
    nFrames = len(functionHistory)
    frames = []
    for i in range(nFrames):
        t, curFunction = functionHistory[i]
        curOutput = curFunction(X, Y)
        frames.append(curOutput)

    allFrameMinimum = min([np.min(frame) for frame in frames])
    allFrameMaximum = max([np.max(frame) for frame in frames])

    allFrameAbsMaximum = max([np.max(np.abs(frame)) for frame in frames])

    cv0 = frames[0]
    im = ax.imshow(cv0, origin='lower', extent = [xmin, xmax, ymin, ymax], cmap=cmap) # Here make an AxesImage rather than contour
    cb = fig.colorbar(im, cax=cax)

    def getTitleForIndex(i: int) -> str:
        titleAppender = title + ", " if title is not None else ""
        return titleAppender + f'Time = {functionHistory[i][0]:.6f}'
    tx = ax.set_title(getTitleForIndex(0))

    # print(frames)

    def animate(i):
        arr = frames[i]

        if "dynamicVs" in kwargs and kwargs.get("dynamicVs") is not None:
            dynamicVs = kwargs.get("dynamicVs")
            # if type(dynamicVs) == 'float':
            
            vmin = np.min([np.min(pastFrame) for pastFrame in frames[max(i - 10, 0):(i+1)]])
            vmax = np.max([np.max(pastFrame) for pastFrame in frames[max(i - 10, 0):(i+1)]])
            # vmin = allFrameMinimum
            # vmax = allFrameMaximum

            if vmin < 0:
                vmin = 1.001 * vmin
            else:
                vmin = 0.999 * vmin

            if vmax < 0:
                vmax = 0.999 * vmax
            else:
                vmax = 1.001 * vmax

            # vmin = -1.05 * allFrameAbsMaximum
            # vmax = 1.05 * allFrameAbsMaximum

            # else:
            #     vmin = 0.95 * allFrameMinimum
            #     vmax = 1.05 * allFrameMaximum
        else:
            vmax     = kwargs.get('vmax') if 'vmax' in kwargs else np.max(arr)
            vmin     = kwargs.get('vmin') if 'vmin' in kwargs else np.min(arr)

        im.set_data(arr)
        im.set_clim(vmin, vmax)
        tx.set_text(getTitleForIndex(i))
        # In this version you don't have to do anything to the colorbar,
        # it updates itself when the mappable it watches (im) changes

    ani = animation.FuncAnimation(fig, animate, frames=nFrames, interval = interval)

    if saveAddr is not None:
        path = os.environ['PYTHONPATH']
        saveTo = f'{path}/{saveAddr}.gif'
        print(f"Saving to {saveTo}")
        ani.save(saveTo, writer='pillow', fps=1000/interval, dpi=80)

    if show:
        plt.show()

if __name__ == "__main__":
    centeredDistributionFunction = lambda s, X, Y: s * np.exp(-(X*X + Y*Y)*s/2) /np.sqrt(2 * np.pi * s)
    functionHistoryToUse = [(
            np.round(0.1*s, 1), 
            (lambda sp: lambda X, Y: centeredDistributionFunction(sp, X, Y))(s)
        ) 
        for s in range(1, 50)
    ]
    

    plotXyVectorizedHeatmapMovieOnGrid(
        functionHistory = functionHistoryToUse,#: List[Tuple[float, Callable[[float, float], float]]], # Callable should expect to receive X and Y as np.array
        xmin = -1,
        xmax = 1,
        ymin = -1,
        ymax = 1,
        nPartitionPerAxis = 500,
        title = "Centered Distribution",
        saveAddr = "testGif",
        show = True,
        interval = 250,
        cmap = 'bwr'
    )


    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # # I like to position my colorbars this way, but you don't have to
    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%', '5%')

    # def f(x, y):
    #     return np.exp(x) + np.sin(y)

    # N = 100
    # x = np.linspace(-3.0, 3.0, N)
    # y = np.linspace(-2.0, 2.0, N)
    # X, Y = np.meshgrid(x, y)

    # # This is now a list of arrays rather than a list of artists
    # nFrames = 50
    # frames = []
    # for i in range(nFrames):
    #     s = i+1
    #     curZ = s*np.exp(-s*(X*X + Y *Y)/2) / (np.sqrt(2 * np.pi * s))
    #     frames.append(curZ)

    # cmap = 'bwr'
    # cv0 = frames[0]
    # im = ax.imshow(cv0, origin='lower', cmap=cmap) # Here make an AxesImage rather than contour
    # cb = fig.colorbar(im, cax=cax)
    
    # titleAppender = title + ", " if title is not None else ""
    # tx = ax.set_title(titleAppender + f'Frame ')

    # def animate(i):
    #     arr = frames[i]
    #     vmax     = np.max(arr)
    #     vmin     = np.min(arr)
    #     im.set_data(arr)
    #     im.set_clim(vmin, vmax)
    #     tx.set_text('Frame {0}'.format(i))
    #     # In this version you don't have to do anything to the colorbar,
    #     # it updates itself when the mappable it watches (im) changes

    # ani = animation.FuncAnimation(fig, animate, frames=nFrames, interval = 500)

    # plt.show()

    # N = 100
    # x = np.linspace(-3.0, 3.0, N)
    # y = np.linspace(-2.0, 2.0, N)

    # X, Y = np.meshgrid(x, y)
    # Z = np.exp(-(X*X + Y *Y)/2) / (np.sqrt(2 * np.pi))

    # fig = plt.figure()

    # ax = fig.add_subplot(111)
    # div = make_axes_locatable(ax)
    # cax = div.append_axes('right', '5%', '5%')

    # im = ax.imshow(Z)    
    # cbar = fig.colorbar(im, cax = cax)
    # tx = ax.set_title('Time = 0')

    # # def init():
    # #     ax.clear()
    # #     cs = ax.contourf(X, Y, Z, locator=mticker.MaxNLocator(), cmap='bwr')      
    #     # cbar = fig.colorbar(cs)

    # # Define the animate function
    # def animate(i):
    #     Z = (i+1)*np.exp(-(i+1)*(X*X + Y *Y)/2) / (np.sqrt(2 * np.pi * (i+1)))
    #     ax.clear()
    #     cax.cla()

    #     cs = ax.contourf(X, Y, Z, locator=mticker.MaxNLocator(), cmap='bwr')      
    #     ax.set_title(f"Time = {i}")


    #     im.set_data(Z)
    #     # im.set_clim(vmin, vmax)

    # ani = animation.FuncAnimation(fig, animate, frames=20, repeat=False, interval = 1000)   
    # plt.show()

    # dimension = (100, 100)
    # fig, ax = plt.subplots(figsize=(3, 3))  # Reduce the figure size
    # cmap = 'bwr'

    # # Create the heatmap with initial data
    # data = np.random.rand(*dimension)
    # heatmap = sns.heatmap(data, ax=ax, vmax=.8, cbar=False, cmap = cmap, annot=False, fmt=".2f")

    # def init():
    #     data = np.zeros(dimension)
    #     ax.clear()
    #     # sns.heatmap(data, ax=ax, vmax=.8, cbar=False, cmap = cmap, annot=False, fmt=".2f")


    # # Define the animate function
    # def animate(i):
    #     data = np.random.rand(*dimension)
    #     ax.clear()
    #     sns.heatmap(data, ax=ax, vmin = 0, vmax=.8, cbar=False, cmap = cmap, annot=False, fmt=".2f")
    #     return ax

    # # Create the animation
    # ani = animation.FuncAnimation(fig, animate, init_func=init, frames=20, repeat=False, interval = 1000)
    # plt.show()

    # # Save the animation as a GIF with optimized parameters
    # # ani.save('animated_heatmap.gif', writer='pillow', fps=2, dpi=80)