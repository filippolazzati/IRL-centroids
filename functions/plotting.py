import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_multiple_policies(
        policies: dict,
        save_fig: bool = False,
        namefig: str = ''
):
    """
    Take in input a dictionary of policies and MDP\Rs, and plot them.
    """

    # get size state space
    n = next(iter(policies.values()))[0].shape[0]
    
    fig, axes = plt.subplots(1, len(policies), figsize=(4 * len(policies), 4),constrained_layout=True)

    if len(policies) == 1:
        axes = [axes]

    # direction deltas: (dx, dy) for plotting the arrows
    dir_map = {
        0: (0.28, 0),   # right
        1: (-0.28, 0),  # left
        2: (0, -0.28),  # up
        3: (0, 0.28),   # down
    }
    
    vmin = 0
    vmax = 1

    pc = [None]*len(policies)

    for idx,(ax,k) in enumerate(zip(axes,policies)):
        pi, d, s0x, s0y, C = policies[k]

        det = True
        if len(pi.shape) == 3:
            det = False

        for i in range(n):
            for j in range(n):
                if det:
                    a = pi[i, j]
                    if a != -1:
                        if a != 4:
                            dx, dy = dir_map[a]
                            # draw arrow from the center of cell (j, i)
                            ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black', linewidth=1)
                        else:
                            ax.scatter(j, i, s=15, color='black', marker='o')  # s controls size
                else:
                    if pi[i,j,0] != -1:  # use -1 to say no policy in state i,j
                        for a in range(4):
                            ax.arrow(j, i, dir_map[a][0]*pi[i,j,a], dir_map[a][1]*pi[i,j,a],
                                     head_width=0.2*pi[i,j,a], head_length=0.2*pi[i,j,a],
                                     fc='black', ec='black', linewidth=1)

                        ax.scatter(j, i, s=15*pi[i,j,4], color='black', marker='o')  # s controls size

        # plot initial state
        rect = patches.Rectangle((s0y - 0.5, s0x - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # plot constraints
        if C is not None:
            for i,j in zip(C[0],C[1]):
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=0.3,
                                         edgecolor='lightgray', facecolor='saddlebrown')
                ax.add_patch(rect)

        ax.xaxis.set_ticks_position('top')
    
        ax.set_title(k, pad=20)

        pc[idx] = ax.imshow(d, cmap='Blues', interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax)
    
    fig.colorbar(pc[0], ax=axes[:], orientation='vertical', shrink=0.8)

    if save_fig:
        fig.savefig('images/'+namefig+".pdf", format="pdf", dpi=1200)

    plt.show()
