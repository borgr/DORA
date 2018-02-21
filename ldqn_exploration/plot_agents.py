import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns

from tensorflow.python.summary.event_accumulator import EventAccumulator

# sns.set(color_codes=True)


def plot_ygrid(magnitude, ymin=None, ymax=None, ax=None, alpha=0.3):
    ax = init_ax(ax)
    ymin, ymax = init_ylim(ymin, ymax, ax)
    # not efficient if far from 0
    i = 1
    while magnitude * i < ymax:
        y = magnitude * i
        i += 1
        if y > ymin:
            plt.axhline(y=y, lw=0.5, color="black",
                        alpha=alpha, linestyle='--')
    i = 0
    print(magnitude, ymin)
    while magnitude * i > ymin:
        y = magnitude * i
        i -= 1
        if y < ymax:
            plt.axhline(y=y, lw=0.5, color="black",
                        alpha=alpha, linestyle='--')


def remove_spines(ax=None):
    ax = init_ax(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def init_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    return ax


def init_ylim(ymin=None, ymax=None, ax=None):
    ax = init_ax(ax)
    if ymin is None or ymax is None:
        tymin, tymax = ax.get_ylim()
        if ymin is None:
            ymin = tymin
        if ymax is None:
            ymax = tymax
    return ymin, ymax


def beautify_heatmap(colorbar=None, magnitude=None, ymin=None, ymax=None,  ax=None, fontsize=14):
    ax = init_ax(ax)
    remove_spines(ax)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    if colorbar:
        colorbar.ax.tick_params(labelsize=fontsize)


def beautify_lines_graph(magnitude, ymin=None, ymax=None, ax=None, fontsize=14, ygrid_alpha=None):
    ax = init_ax(ax)
    remove_spines(ax)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary
    # chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # # Limit the range of the plot to only where the data is.
    # # Avoid unnecessary whitespace.
    # plt.ylim(ymin, ymax)
    # plt.xlim(xmin, xmax)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    # Remove the tick marks; they are unnecessary with the tick lines we
    # just plotted.
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    data = {"ymin": ymin, "ymax": ymax,
            "magnitude": magnitude, "alpha": ygrid_alpha, "ax": ax}
    data = dict((k, v) for k, v in data.items() if v is not None)
    # plot_ygrid(**data)


def add_legend(ax=None, loc='best', fancybox=True, fontsize=14, shadow=True):
    ax = init_ax(ax)
    ax.legend(loc=loc, fancybox=fancybox, fontsize=fontsize, shadow=shadow)


def many_colors(labels, colors=cm.rainbow):
    """creates colors, each corresponding to a unique label

    use for a list of colors:
    example = [(230, 97, 1), (253, 184, 99),
                   (178, 171, 210), (94, 60, 153)]
    for i in range(len(example)):
        r, g, b = example[i]
        example[i] = (r / 255., g / 255., b / 255.)

     places with colors
     https://matplotlib.org/users/colormaps.html
     http://colorbrewer2.org/#type=diverging&scheme=PuOr&n=4
     http://tableaufriction.blogspot.co.il/2012/11/finally-you-can-use-tableau-data-colors.html
     https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/"""
    cls = set(labels)
    if len(cls) == 2:
        return dict(zip(cls, ("blue", "orange")))
    return dict(zip(cls, colors(np.linspace(0, 1, len(cls)))))


def export_tensorflow_log(path, out):
    print("exporting", path)
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 0,
        'images': 0,
        'scalars': 100000,
        'histograms': 0
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    # print(event_acc.Tags())

    # step =   event_acc.Scalars('steps')
    scores = event_acc.Scalars('episode/score')
    # steps = event_acc.Scalars('episode/steps')
    # assert len(steps) == len(score)
    res = []
    for score in scores:
        res.append(tuple(score))
    # print("res", res)
    # print("will be written to", out)
    # return
    with open(out, "w") as fl:
        json.dump(res, fl)
    return res


def extract_data(runs_dirs, jsons_dir, force):
    for i, runs in enumerate(runs_dirs):
        for root, drs, files in os.walk(runs):
            for dr in drs:
                outfile = None
                dr = os.path.join(root, dr)
                if "way_c" in root.lower() and dr[-4:].isdigit():
                    outfile = os.path.join(jsons_dir, str(
                        i) + "freeway_c" + dr.split(os.sep)[-1])
                elif "way_dqn" in root.lower() and dr[-4:].isdigit():
                    outfile = os.path.join(jsons_dir, str(
                        i) + "freeway_dqn" + dr.split(os.sep)[-1])
                elif "way_e" in root.lower() and dr[-4:].isdigit():
                    outfile = os.path.join(jsons_dir, str(
                        i) + "freeway_e" + dr.split(os.sep)[-1])
                if outfile:
                    if force or not os.path.isfile(outfile):
                        export_tensorflow_log(dr, outfile)
                    else:
                        print(outfile + " already exists")


def main():
    jsons_dir = r"/cs/labs/oabend/borgr/ldqn_exploration/jsons"
    runs_dirs = ["/cs/labs/oabend/borgr/ldqn_exploration/atari-rl/runs/",
                 "/cs/labs/oabend/borgr/ldqn_exploration/atari-rl/atari-rl/runs/"]
    tb_data=True
    # force = True
    # extract_data(runs_dirs, jsons_dir, force)
    # return
    # agents = {"Density": ["run-freeway_c-.-tag-episode-score.json"],
    #           "DQN": ["run-freeway_dqn-.-tag-episode-score.json"],
    #           "$E$-values": ["run-freeway_e-.-tag-episode-score.json"]}
    xlim_up = 3.4
    DENSITY = "Density"
    DQN = "DQN"
    E = "$E$-values"
    agents = {DENSITY: [], DQN: [], E: []}
    for root, dr, files in os.walk(jsons_dir):
        for filename in files:
            if filename[0].isdigit() == (tb_data and "way_c" not in filename.lower() and "way_dqn" not in filename.lower()):
                continue
            filename = os.path.join(root, filename)
            if "way_c" in filename.lower():
                print("c", filename)
                agents[DENSITY].append(filename)
            if "way_dqn" in filename.lower():
                print("dqn", filename)
                agents[DQN].append(filename)
            if "way_e" in filename.lower():
                print("e", filename)
                agents[E].append(filename)
    # print([len(x) for x in agents.values()])
    # return
    # colorbrewer = [(230, 97, 1), (253, 184, 99),
    #                (178, 171, 210), (94, 60, 153)]
    # for i in range(len(colorbrewer)):
    #     r, g, b = colorbrewer[i]
    #     colorbrewer[i] = (r / 255., g / 255., b / 255.)
    # colors = many_colors(agents.keys(),
    #                      mpl.colors.ListedColormap(colorbrewer))
    colors = sns.color_palette("coolwarm", 5)
    print(colors)
    colors = {E:"black", DQN: colors[0], DENSITY:colors[-1]}
    ts_data = []
    scale = 1000000
    for agent, agent_runs in agents.items():
        if agent != DENSITY:
            continue
        xs = []
        ys = []
        for run_file in agent_runs:
            with open(os.path.join(root, run_file)) as fl:
                run = json.load(fl)
            x = np.insert(np.array([tpl[1] for tpl in run]), 0, 0)
            y = np.insert(np.array([tpl[2] for tpl in run]), 0, 0)
            # for i in range(len(x)):
            #     ts_data.append((x[i], y[i], agent))
            # print(x[-1] / scale, y[-1], agent)
            xs.append(x)
            ys.append(y)
        agent_x = np.arange(0, min(max([x[-1] for x in xs]), xlim_up * scale), 100)
        agent_ys = [[] for i in range(len(agent_x))]
        for x, y in zip(xs, ys):
            print(agent, x[-1])
            last = 0
            mx_x = x[-1]
            for i, cur_x in enumerate(agent_x):
                if cur_x > mx_x:
                    break
                last += np.argmax(x[last:] >= cur_x)
                if x[last] == cur_x:
                    cur_y = y[last]
                else:
                    ya = y[last - 1]
                    yb = y[last]
                    xa = x[last - 1]
                    xb = x[last]
                    cur_y = ya + (xb - cur_x) * ((ya - yb) / (xa - xb))
                agent_ys[i].append(cur_y)
                # print(agent_ys[-1])
                assert agent_ys[
                    i][-1] >= 0, (agent_ys[i][-1], xa, xb, ya, yb, cur_x)
        # for i,(x, y) in enumerate(zip(xs, ys)):
        #     print(i)
        #     plt.plot(x / scale, y, label=agent, color=colors[agent])
        #     plt.show()
        print([tmp for tmp, tmpx in zip(agent_ys, agent_x) if 2700000<tmpx <3000000], len(agent_ys))
        # return
        y = []
        stds = []
        for cur_y in agent_ys:
            if len(cur_y) > 0:
                y.append(np.mean(cur_y))
                stds.append(np.std(cur_y))
        x = agent_x
        # for i in range(len(agent_ys)):
        #     for ts_y in agent_ys[i]:
        #         ts_data.append((x[i], ts_y, agent))
        # print(agent, [len(cur) for cur in agent_ys])
        y = [tmp for tmp, tmpx in zip(y, x) if 2700000<tmpx <3000000]
        stds = [tmp for tmp, tmpx in zip(stds, x) if 2700000<tmpx <3000000]
        stds = np.array(stds)
        print(list(zip(y[-10:], stds[-10:])))
        x = np.array([tmpx for tmpx in x if 2700000<tmpx <3000000])
        y = np.array(y)
        plt.plot(x / scale, y, label=agent, color=colors[agent])
        plt.fill_between(x / scale, y + stds, np.maximum(np.zeros(y.shape), y - stds), alpha=.3, color=colors[agent], linewidth=0)
        print("calculated for", agent)
    # ts_data = pd.DataFrame(ts_data, columns=["Million steps","Last episode score", "agent"])
    # ts_data["unit"] = ts_data.groupby(['Million steps','Last episode score']).cumcount()
    # print(ts_data)
    # ax = sns.tsplot(data=ts_data, time="Million steps", unit="unit",
    #        condition="agent", value="Last episode score", ci="sd")
    # plt.show()
    plt.xlim(0, xlim_up)
    # beautify_lines_graph(5, ygrid_alpha=0.5)
    # add_legend()
    plt.xlabel("Million steps")
    plt.ylabel("Last episode score")
    plt.savefig(filename="/cs/usr/borgr/Desktop/lovlior.png", bbox_inches="tight", dpi=400)
    plt.show()

if __name__ == '__main__':
    main()