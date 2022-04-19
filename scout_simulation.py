from field_setup import Field
from flower_finder import scoutSwarm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc, rcParams
from matplotlib.lines import Line2D
rcParams['animation.embed_limit'] = 2**128

time_data = []
percent_found = []
def animate(i, swarm):
    time_data.append(i)

    swarm.iterate()
    line.set_data(swarm.agents.T[0], swarm.agents.T[1])

    unfound_f = swarm.map.unfound_flowers
    found_f = swarm.map.found_flowers

    percent_found.append((len(found_f)/(len(found_f)+len(unfound_f)))*100)

    perc_line.set_data(time_data, percent_found)

    unfound_line.set_data(unfound_f.T[0], unfound_f.T[1])
    found_line.set_data(found_f.T[0], found_f.T[1])

    return line, unfound_line, found_line, perc_line


if __name__ == '__main__':
    x_lim = 100
    y_lim = 100

    f = Field(x_lim, y_lim, 20, 200)
    s = scoutSwarm(20, f)
    s.spawn_agents()

    fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(16,8), dpi=70, facecolor='w', edgecolor='k')

    ax1.set_xlim(0, x_lim)
    ax1.set_ylim(0, y_lim)
    #plt.scatter(s.agents[:,0], s.agents[:,1], color='k')
    #plt.close()
    #ax1.scatter(s.waypoints[:, 0], s.waypoints[:, 1], marker='x', color = 'k')

    ax2.set_xlim((0, 1200))
    ax2.set_ylim((0, 100))
    ax2.set_yticks(np.arange(0, 100, 10))
    ax2.grid()

    fontsize = 12
    ax2.set_xlabel('Time', fontsize=fontsize)
    ax2.set_ylabel('Percentage of Flowers Found (%)', fontsize=fontsize)

    global line, unfound_line, found_line, perc_line
    bee_marker = "$\u29DE$"
    line, = ax1.plot([], [], marker=bee_marker, markersize=10, color='k', linewidth=0)

    flower_marker = "$\u273F$"
    unfound_line, = ax1.plot([], [], marker=flower_marker, markersize=10, color='r', linewidth=0)
    found_line, = ax1.plot([], [], marker=flower_marker, markersize=10, color='g', linewidth=0)
    perc_line, = ax2.plot([], [], 'g-', markersize=5)

    legend_markers = [Line2D([], [], marker=bee_marker, markersize=10, color='k', linewidth=0, zorder=2),
                      Line2D([], [], marker=flower_marker, markersize=10, color='r', linewidth=0, zorder=1),
                      Line2D([], [], marker=flower_marker, markersize=10, color='g', linewidth=0, zorder=1)]

    ax1.legend(legend_markers, ['Scout Drone', 'Unfound', 'Found'], ncol=3, loc='upper left', bbox_to_anchor=(0.15, 1.06))

    def init():
        line.set_data([], [])
        unfound_line.set_data([], [])
        found_line.set_data([], [])
        perc_line.set_data([], [])
        return line, unfound_line, found_line, perc_line,

    anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(s,),
                                   frames=1200, interval=200, blit=True, cache_frame_data=False)

    #plt.show()
    anim.save('scout_sim_animation_legend2.mp4', fps=25, dpi=200)










