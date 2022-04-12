from field_setup import Field
from flower_finder import scoutSwarm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc, rcParams
from matplotlib.font_manager import FontProperties
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

    f = Field(x_lim, y_lim, 50, 300)
    s = scoutSwarm(20, f)
    s.spawn_agents()

    fig, ax1 = plt.subplots(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    ax1.set_xlim(0, x_lim)
    ax1.set_ylim(0, y_lim)
    #plt.scatter(s.agents[:,0], s.agents[:,1], color='k')
    #plt.close()
    plt.scatter(s.waypoints[:, 0], s.waypoints[:, 1], marker='x', color = 'k')

    global line, unfound_line, found_line
    bee_marker = "$\u29DE$"
    line, = ax1.plot([], [], marker=bee_marker, markersize=10, color='k', linewidth=0)

    flower_marker = "$\u273F$"
    unfound_line, = ax1.plot([], [], marker=flower_marker, markersize=10, color='r', linewidth=0)
    found_line, = ax1.plot([], [], marker=flower_marker, markersize=10, color='g', linewidth=0)

    def init():
        line.set_data([], [])
        unfound_line.set_data([], [])
        found_line.set_data([], [])
        return line, unfound_line, found_line,

    anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(s,),
                                   frames=1500, interval=200, blit=True, cache_frame_data=False)

    plt.show()
    #anim.save('sim_animation.mp4', fps=25, dpi=200)










