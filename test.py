import random
import numpy as np
import matplotlib.pyplot as plt
from environment import Environ


np.random.seed(0)
random.seed(0)


def plottrack(trackmap):
    _, ax = plt.subplots(figsize=(10, 16))  # (10, 16)
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
    # plt.show()
    return ax


# 设置赛道
# track1
trackmap = {
    "left side x": np.array(
        [3] * 3 + [2] * 7 + [1] * 8 + [0] * 10 + [1, 2] + list(range(2, 18))
    ),
    "left side y": np.array(list(range(33)) + [32] * 13),
    "right side x": np.array([10] * 25 + list(range(11, 18))),
    "right side y": np.array(list(range(25)) + [25] * 7),
}


# 实验一：衰减常数c的选取
c = [0.8, 0.9, 0.99, 1]
i = 0
agent_names = [
    {
        "name": "policy-gradient (decay-epsilon greedy)",
        "trackmap": trackmap,
        "alphax": c[i],
        "eps": 0.5,
        "s_max": 16,
        "g_win": 2,
        "g_reset": -10,
    },
]

# softmax算法的f()需要在agent.py文件中将f(n)设置为log(n+10)，设置的位置已标注了注释f(n)
# agent_names = [
#     {
#         "name": "policy-gradient (softmax)",
#         "trackmap": trackmap,
#         "alphax": c[i],
#         "s_max": 16,
#         "g_win": 2,
#         "g_reset": -10,
#     },
# ]

env = Environ(trackmap, agent_names)
env.simulation()
times = env.evaluate_time()
np.save(file="./data/times.npy", arr=times)
reset_times = env.evaluate_resetfreq()
np.save(file="./data/reset times.npy", arr=reset_times)


# 实验二：随机性eps0, f()的选取
eps0 = [0.1, 0.5, 0.9]
i = 0
agent_names = [
    {
        "name": "policy-gradient (decay-epsilon greedy)",
        "trackmap": trackmap,
        "alphax": 0.9,
        "eps": eps0[i],
        "s_max": 16,
        "g_win": 2,
        "g_reset": -10,
    },
]

# softmax算法的f()需要在agent.py文件中设置，设置的位置已标注了注释f(n)
# agent_names = [
#     {
#         "name": "policy-gradient (softmax)",
#         "trackmap": trackmap,
#         "alphax": 0.9,
#         "s_max": 16,
#         "g_win": 2,
#         "g_reset": -10,
#     },
# ]

env = Environ(trackmap, agent_names)
env.simulation()
times = env.evaluate_time()
np.save(file="./data/times.npy", arr=times)
reset_times = env.evaluate_resetfreq()
np.save(file="./data/reset times.npy", arr=reset_times)


# 实验三：上界s_max的选取
s_max = [12, 16, 20]
i = 0
agent_names = [
    {
        "name": "policy-gradient (decay-epsilon greedy)",
        "trackmap": trackmap,
        "alphax": 0.9,
        "eps": 0.1,
        "s_max": s_max[i],
        "g_win": 2,
        "g_reset": -10,
    },
]

# softmax算法的f()需要在agent.py文件中将f(n)设置为(n+10)^2，设置的位置已标注了注释f(n)
# agent_names = [
#     {
#         "name": "policy-gradient (softmax)",
#         "trackmap": trackmap,
#         "alphax": 0.9,
#         "s_max": s_max[i],
#         "g_win": 2,
#         "g_reset": -10,
#     },
# ]

env = Environ(trackmap, agent_names)
env.simulation()
times = env.evaluate_time()
np.save(file="./data/times.npy", arr=times)
reset_times = env.evaluate_resetfreq()
np.save(file="./data/reset times.npy", arr=reset_times)


# 实验四：增益g_win的选取
g_win = [0, 2, 5]
i = 0
agent_names = [
    {
        "name": "policy-gradient (decay-epsilon greedy)",
        "trackmap": trackmap,
        "alphax": 0.9,
        "eps": 0.1,
        "s_max": 16,
        "g_win": g_win[i],
        "g_reset": -10,
    },
]

# softmax算法的f()需要在agent.py文件中将f(n)设置为(n+10)^2，设置的位置已标注了注释f(n)
# agent_names = [
#     {
#         "name": "policy-gradient (softmax)",
#         "trackmap": trackmap,
#         "alphax": 0.9,
#         "s_max": 16,
#         "g_win": g_win[i],
#         "g_reset": -10,
#     },
# ]

env = Environ(trackmap, agent_names)
env.simulation()
times = env.evaluate_time()
np.save(file="./data/times.npy", arr=times)
reset_times = env.evaluate_resetfreq()
np.save(file="./data/reset times.npy", arr=reset_times)


# 实验四：增益g_reset的选取
g_reset = [0, -5, -10, -15, -20, -30]
i = 0
agent_names = [
    {
        "name": "policy-gradient (decay-epsilon greedy)",
        "trackmap": trackmap,
        "alphax": 0.9,
        "eps": 0.1,
        "s_max": 16,
        "g_win": 2,
        "g_reset": g_reset[i],
    },
]

# softmax算法的f()需要在agent.py文件中将f(n)设置为(n+10)^2，设置的位置已标注了注释f(n)
# agent_names = [
#     {
#         "name": "policy-gradient (softmax)",
#         "trackmap": trackmap,
#         "alphax": 0.9,
#         "s_max": 16,
#         "g_win": 2,
#         "g_reset": g_reset[i],
#     },
# ]

env = Environ(trackmap, agent_names)
env.simulation()
times = env.evaluate_time()
np.save(file="./data/times.npy", arr=times)
reset_times = env.evaluate_resetfreq()
np.save(file="./data/reset times.npy", arr=reset_times)


# 最优参数下的算法比较，并绘制行车轨迹
# 请在agent.py文件中的构造函数中自行设定w, theta的取值，示例的代码行已注释
agent_names = [
    {
        "name": "policy-gradient (decay-epsilon greedy)",
        "trackmap": trackmap,
        "alphax": 0.9,
        "eps": 0.1,
        "s_max": 16,
        "g_win": 2,
        "g_reset": -20,
        "if_record": True,
        "alpha": np.array([0, 0]),
    },
]

# softmax算法的f()需要在agent.py文件中将f(n)设置为(n+10)^2，设置的位置已标注了注释f(n)
# agent_names = [
#     {
#         "name": "policy-gradient (softmax)",
#         "trackmap": trackmap,
#         "alphax": 0.9,
#         "s_max": 16,
#         "g_win": 2,
#         "g_reset": -10,
#         "if_record": True,
#         "alpha": np.array([0, 0]),
#     },
# ]

# 绘制赛道
plt.ion()
ax = plottrack(trackmap)
env = Environ(trackmap, agent_names, ax, epi=30, simu=1)
s, piv = env.simulation()
s = np.array(s[0])
piv = np.array(piv[0])
np.save(file="./data/s.npy", arr=s)
np.save(file="./data/piv.npy", arr=piv)
times = env.evaluate_time()
reset_times = env.evaluate_resetfreq()


# 新赛道
# track2
trackmap = {
    "left side x": np.array(
        [0] * 3
        + list(range(1, 14))
        + [14] * 5
        + [13, 12]
        + [11] * 4
        + list(range(12, 33))
    ),
    "left side y": np.array(list(range(30)) + [29] * 2 + [30] * 16),
    "right side x": np.array([24] * 17 + list(range(25, 33))),
    "right side y": np.array(list(range(18)) + [17, 18] + [19] * 3 + [20] * 2),
}
# softmax算法的f()需要在agent.py文件中将f(n)设置为(n+10)^2，设置的位置已标注了注释f(n)
agent_names = [
    {
        "name": "policy-gradient (decay-epsilon greedy)",
        "trackmap": trackmap,
        "g_reset": -20,
    },
    {
        "name": "policy-gradient (softmax)",
        "trackmap": trackmap,
        "g_reset": -10,
    },
]
env = Environ(trackmap, agent_names)
env.simulation()
times = env.evaluate_time()
np.save(file="./data/times.npy", arr=times)
reset_times = env.evaluate_resetfreq()
np.save(file="./data/reset times.npy", arr=reset_times)
