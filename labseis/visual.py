import numpy as np
import matplotlib.pyplot as plt

def wiggles(xt, t_axis, offset, ax=None, fig=None, scalebar=True, scale=0.5, scale_range='trace', legends=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=(10, 5))
    normalized = np.max(xt)

    if scale_range=='trace':
        normalized = np.max(xt, axis=-1)

    elif scale_range == 'all':
        normalized = np.max(xt, axis=-1)
        normalized[normalized<np.max(normalized)] = np.max(normalized)

    for i in range(xt.shape[0]):
        if legends is not None:
            ax.plot(t_axis, scale*xt[i]/normalized[i] + offset[i],label=legends[i])
        else:
            ax.plot(t_axis, scale*xt[i]/normalized[i] + offset[i])


    if scalebar:
        for i in range(xt.shape[0]):
            ax.plot([t_axis[0],t_axis[0]], [offset[i],offset[i]+scale], 'k')
            ax.annotate('{}'.format(normalized[i]), xy=[0,offset[i]+0.1])

    plt.legend(loc = 'upper right')
    # if legends is not None:
    #     for i in range(xt.shape[0]):
    #         ax.annotate('{}'.format(legends[i]), xy=[t_axis[-1],offset[i]+0.1])
    
    return fig, ax