from field_setup import Field
from tsp_pollination import pollenSwarm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc, rcParams
from matplotlib.lines import Line2D
rcParams['animation.embed_limit'] = 2**128


def animate(i, swarm):
    swarm.iterate()
    line.set_data(swarm.agents.T[0], swarm.agents.T[1])

    unfound_f = swarm.map.unfound_flowers
    found_f = swarm.map.found_flowers

    unfound_line.set_data(unfound_f.T[0], unfound_f.T[1])
    found_line.set_data(found_f.T[0], found_f.T[1])

    return line, unfound_line, found_line,


if __name__ == '__main__':
    x_lim = 100
    y_lim = 100

    f = Field(x_lim, y_lim, 20, 200)
    s = pollenSwarm(15, f)
    s.spawn_agents()

    fig, ax1 = plt.subplots(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

    route = s.routes[0]
    xs = []
    ys = []
    for p in route:
        xs.append(p.x)
        ys.append(p.y)
    xs.append(xs[0])
    ys.append(ys[0])
    plt.plot(xs, ys, 'k--')

    ax1.set_xlim(0, x_lim)
    ax1.set_ylim(0, y_lim)

    global line, unfound_line, found_line
    bee_marker = "$\u29DE$"
    line, = ax1.plot([], [], marker=bee_marker, markersize=10, color='k', linewidth=0, zorder=2)

    flower_marker = "$\u273F$"
    unfound_line, = ax1.plot([], [], marker=flower_marker, markersize=10, color='r', linewidth=0, zorder=1)
    found_line, = ax1.plot([], [], marker=flower_marker, markersize=10, color='g', linewidth=0, zorder=1)

    legend_markers = [Line2D([], [], marker=bee_marker, markersize=10, color='k', linewidth=0, ),
                      Line2D([], [], color='k', ls='--'),
                      Line2D([], [], marker=flower_marker, markersize=10, color='g', linewidth=0,)]

    ax1.legend(legend_markers, ['Pollination Drone', 'Route', 'Target'], ncol=3, loc='upper left',
               bbox_to_anchor=(0.24, 1.06))

    def init():
        line.set_data([], [])
        unfound_line.set_data([], [])
        found_line.set_data([], [])
        return line, unfound_line, found_line,


    anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(s,),
                                   frames=1500, interval=200, blit=True, cache_frame_data=False)

    #plt.show()
    anim.save('pollination_sim_animation.mp4', fps=25, dpi=200)




