import numpy as np
import matplotlib.pyplot as plt


def result_plot(agent, filenames, legends, irange=range(2), alpha1=0.5, alpha2=0.2):
    """
    agent: decay epsilon greedy, softmax
    """

    times, reset_times = [], []
    if agent:
        for name in filenames:
            times.append(np.load(file="./" + agent + "/" + name + "/times.npy"))
            reset_times.append(
                np.load(file="./" + agent + "/" + name + "/reset times.npy")
            )
    else:
        for name in filenames:
            times.append(np.load(file=name + "/times.npy"))
            reset_times.append(np.load(file=name + "/reset times.npy"))

    _, ax = plt.subplots()
    epi = times[0].shape[1]
    for i in range(len(filenames)):
        ax.fill_between(
            range(epi),
            np.percentile(times[i][0, :, :], 75, axis=1),
            np.min(times[i][0, :, :], axis=1),
            alpha=alpha2,
            linewidth=0,
        )
        ax.plot(
            np.percentile(times[i][0, :, :], 50, axis=1),
            "-",
            label=legends[i],
        )
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Times")
    plt.show()

    _, ax = plt.subplots()
    epi = reset_times[0].shape[1]
    for i in irange:
        ax.fill_between(
            range(epi),
            np.percentile(reset_times[i][0, :, :], 75, axis=1),
            np.min(reset_times[i][0, :, :], axis=1),
            alpha=alpha1,
            linewidth=0,
        )
        ax.plot(
            np.percentile(reset_times[i][0, :, :], 50, axis=1),
            "-",
            label=legends[i],
        )
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reset times")
    plt.show()

    _, ax = plt.subplots()
    epi = reset_times[0].shape[1]
    for i in range(len(filenames)):
        ax.plot(
            np.sum(reset_times[i][0, :, :] != 0, axis=1),
            "-",
            label=legends[i],
        )
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("% Reset")
    plt.show()


# alpha
filenames = ["alpha x 0.8", "alpha x 0.9", "alpha x 0.99", "alpha x 1"]
legends = filenames
result_plot("decay epsilon greedy", filenames, legends)
result_plot("softmax", filenames, legends)

# epsilon
filenames = ["eps=0.1", "alpha x 0.9", "eps=0.9"]
legends = ["eps=0.1", "eps=0.5", "eps=0.9"]
result_plot("decay epsilon greedy", filenames, legends, irange=range(3), alpha1=0.2)

# f(n)
filenames = ["log(sqrt)", "alpha x 0.9", "sqrt", "()", "()^2", "()^3"]
legends = ["log(sqrt())", "log()", "sqrt()", "()", "()^2", "()^3"]
result_plot("softmax", filenames, legends, irange=range(2, 6))

# s_max
filenames = ["s limit=12", "eps=0.1", "s limit=20"]
legends = [r"$s_{max}=12$", r"$s_{max}=16$", r"$s_{max}=20$"]
result_plot("decay epsilon greedy", filenames, legends)
filenames = ["s limit=12", "()^2", "s limit=20"]
result_plot("softmax", filenames, legends)

# g_win
filenames = ["win=0", "eps=0.1", "win=5"]
legends = [r"$g_{win}=0$", r"$g_{win}=2$", r"$g_{win}=5$"]
result_plot("decay epsilon greedy", filenames, legends)
filenames = ["win=0", "()^2", "win=5"]
result_plot("softmax", filenames, legends)

# g_reset
filenames = [
    "reset=0",
    "reset=-5",
    "eps=0.1",
    "reset=-15",
    "reset=-20",
    "reset=-30",
]
legends = [
    r"$g_{reset}=0$",
    r"$g_{reset}=-5$",
    r"$g_{reset}=-10$",
    r"$g_{reset}=-15$",
    r"$g_{reset}=-20$",
    r"$g_{reset}=-30$",
]
result_plot("decay epsilon greedy", filenames, legends, irange=range(2, 6))
filenames = [
    "reset=0",
    "reset=-5",
    "()^2",
    "reset=-15",
    "reset=-20",
    "reset=-30",
]
result_plot("softmax", filenames, legends, irange=range(2, 6))

# agents
filenames = ["./decay epsilon greedy/reset=-20", "./softmax/()^2"]
legends = ["policy-gradient (decay-epsilon greedy)", "policy-gradient (softmax)"]
result_plot(None, filenames, legends, alpha2=0.5)


# 绘制各变量值(s0, s1, s2, s3, s4, s5, v, piv1, piv2)在赛道上的分布
trackmap = {
    "left side x": np.array(
        [3] * 3 + [2] * 7 + [1] * 8 + [0] * 10 + [1, 2] + list(range(2, 18))
    ),
    "left side y": np.array(list(range(33)) + [32] * 13),
    "right side x": np.array([10] * 25 + list(range(11, 18))),
    "right side y": np.array(list(range(25)) + [25] * 7),
}


def plot_s(trackmap, s, vtype=None):
    _, ax = plt.subplots(figsize=(5, 16))
    ax.plot(trackmap["left side x"], trackmap["left side y"], "k")
    ax.plot(trackmap["right side x"], trackmap["right side y"], "k")
    ax.set_xticks(range(-1, trackmap["right side x"][-1] + 2))
    ax.set_yticks(range(-1, trackmap["left side y"][-1] + 2))
    plt.hlines(
        0, xmin=trackmap["left side x"][0], xmax=trackmap["right side x"][0], color="r"
    )
    plt.vlines(
        trackmap["right side x"][-1],
        ymin=trackmap["right side y"][-1],
        ymax=trackmap["left side y"][-1],
        color="r",
    )
    plt.text(trackmap["left side x"][0] + 1, -1, "Starting line")
    plt.text(
        trackmap["right side x"][-1] + 0.3,
        trackmap["left side y"][-1] - 4,
        "Finish \n line",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    for key, value in s.items():
        key = eval(key)
        if vtype == "s0s1":
            ax.scatter(
                key[0],
                key[1],
                marker=">",
                c="tab:blue",
                s=(value[-1][0] + 17) * 3,
                alpha=0.5,
                edgecolors="none",
            )
            ax.scatter(
                key[0],
                key[1],
                marker="^",
                c="tab:red",
                s=(value[-1][1] + 17) * 3,
                alpha=0.5,
                edgecolors="none",
            )
        elif vtype == "s2s3":
            ax.scatter(
                key[0],
                key[1],
                marker=">",
                c="tab:blue",
                s=(value[-1][2] + 4) * 30,
                alpha=0.5,
                edgecolors="none",
            )
            ax.scatter(
                key[0],
                key[1],
                marker="^",
                c="tab:red",
                s=(value[-1][3] + 4) * 30,
                alpha=0.5,
                edgecolors="none",
            )
        elif vtype == "s4s5":
            ax.scatter(
                key[0],
                key[1],
                marker=">",
                c="tab:blue",
                s=(value[-1][4] + 1) * 20,
                alpha=0.5,
                edgecolors="none",
            )
            ax.scatter(
                key[0],
                key[1],
                marker="^",
                c="tab:red",
                s=(value[-1][5] + 1) * 30,
                alpha=0.5,
                edgecolors="none",
            )
        elif vtype == "v":
            ax.scatter(
                key[0],
                key[1],
                c="tab:green",
                s=(value[-1][0] + 5) * 15,
                alpha=0.5,
                edgecolors="none",
            )
        elif vtype == "piv1piv2":
            ax.scatter(
                key[0],
                key[1],
                marker=">",
                c="tab:blue",
                s=(value[-1][1]) * 25,
                alpha=0.5,
                edgecolors="none",
            )
            ax.scatter(
                key[0],
                key[1],
                marker="^",
                c="tab:red",
                s=(value[-1][2]) * 25,
                alpha=0.5,
                edgecolors="none",
            )
    plt.show()


s = np.load(file="./data/s.npy", allow_pickle=True)
s = s[()]
piv = np.load(file="./data/piv.npy", allow_pickle=True)
piv = piv[()]
plot_s(trackmap, s, "s0s1")
plot_s(trackmap, s, "s2s3")
plot_s(trackmap, s, "s4s5")
plot_s(trackmap, piv, "v")
plot_s(trackmap, piv, "piv1piv2")


reset_times = np.load(file="./track2/reset times.npy")
legends = ["policy-gradient (decay-epsilon greedy)", "policy-gradient (softmax)"]
_, ax = plt.subplots()
for i in range(2):
    ax.plot(
        np.sum(reset_times[i, :, :] != 0, axis=1),
        "-",
        label=legends[i],
    )
plt.legend()
plt.xlabel("Episode")
plt.ylabel("% Reset")
plt.show()
