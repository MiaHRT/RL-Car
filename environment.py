import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from tqdm import tqdm
import copy


class Environ(object):
    """
    trackmap: map of the track
    agent_names: list of agents
    N: number of agents
    epi: number of episodes in each simulation
    e: current episode
    simu: number of simulations
    state0: list of agents' current states
    speed: list of different agents' speeds
    state1: list of agents' next states
    r: list of rewards
    t: number of steps per episode
    reset_freq: number of resets per episode

    reset(): initializing state when reset
    referee(): determining whether the car has hit the boundary of the track
    """

    def __init__(self, trackmap, agent_names, ax=None, epi=100, simu=100):
        self.trackmap = trackmap
        self.agent_names = agent_names
        self.ax = ax
        self.N = len(agent_names)
        self.epi = epi
        self.__e = 0
        self.simu = simu

        self.t = np.zeros((self.N, epi, simu))
        self.reset_freq = np.zeros((self.N, epi, simu))
        self.__color = plt.get_cmap("GnBu")(np.linspace(0.5, 1, simu))

    # def __del__(self):
    #     print("environment will be deleted!")

    def simulation(self):
        mintime = np.inf * np.ones((self.N))
        bestw, besttheta = [[] for _ in range(self.N)], [[] for _ in range(self.N)]
        for iter in range(self.simu):
            agent = [Agent(**name) for name in self.agent_names]
            s, piv = [], []
            for i in range(self.N):
                for self.__e in tqdm(
                    range(self.epi),
                    desc=f"Simulation {iter}",
                    ncols=80,
                    leave=False,
                    dynamic_ncols=False,
                ):
                    w, theta = agent[i].get_w_theta()
                    state0 = self.reset()
                    # while 1:
                    for _ in range(500):
                        action = agent[i].selection(copy.deepcopy(state0))
                        r = -1
                        state1 = copy.deepcopy(state0)
                        if np.random.random() < 0.1:
                            state1["position"][0] = (
                                state0["position"][0] + state0["speed"][0]
                            )
                            state1["position"][1] = (
                                state0["position"][1] + state0["speed"][1]
                            )
                        else:
                            state1["speed"][0] = state0["speed"][0] + action[0]
                            state1["speed"][1] = state0["speed"][1] + action[1]
                            state1["position"][0] = (
                                state0["position"][0] + state1["speed"][0]
                            )
                            state1["position"][1] = (
                                state0["position"][1] + state1["speed"][1]
                            )
                        tmp = self.referee(state0["position"], state1["position"])
                        agent[i].update(state0, r, state1, tmp)
                        # print(action, state0["position"], state1["position"], tmp)
                        if self.ax:
                            plt.title(
                                "episode={}, times={}".format(
                                    self.__e, self.t[i, self.__e, iter]
                                )
                            )
                            self.ax.plot(
                                [state0["position"][0], state1["position"][0]],
                                [state0["position"][1], state1["position"][1]],
                                color=self.__color[iter],
                            )
                            self.ax.plot(
                                state0["position"][0],
                                state0["position"][1],
                                "o",
                                color=self.__color[iter],
                            )
                            self.ax.plot(
                                state1["position"][0],
                                state1["position"][1],
                                "o",
                                color="orange",
                            )
                            plt.pause(0.01)
                            self.ax.plot(
                                state1["position"][0],
                                state1["position"][1],
                                "o",
                                color=self.__color[iter],
                            )
                        self.t[i, self.__e, iter] += 1
                        if tmp == "win":
                            if self.t[i, self.__e, iter] <= mintime[i]:
                                mintime[i] = self.t[i, self.__e, iter]
                                bestw[i], besttheta[i] = w, theta
                            break
                        elif tmp == "reset":
                            self.reset_freq[i, self.__e, iter] += 1
                            state0 = self.reset()
                        else:
                            state0 = copy.deepcopy(state1)
                        del state1

                tmp1, tmp2 = agent[i].get_s_piv()
                s.append(tmp1)
                piv.append(tmp2)

        if self.ax:
            plt.title(None)
            plt.savefig("./image/agent_trail.png")
        print("min time = ", mintime)
        print("best w = {}, best theta = {}".format(bestw, besttheta))

        return s, piv

    def reset(self):
        state0 = {
            "position": [
                np.random.randint(
                    self.trackmap["left side x"][0] + 1,
                    self.trackmap["right side x"][0],
                ),
                0,
            ],
            "speed": np.array([0, 0]),
        }
        return state0

    def referee(self, state0, state1):
        # win condition
        if state1[0] >= self.trackmap["left side x"][-1]:
            y = state0[1] + (state1[1] - state0[1]) / (state1[0] - state0[0]) * (
                self.trackmap["left side x"][-1] - state0[0]
            )
            if (
                y > self.trackmap["right side y"][-1]
                and y < self.trackmap["left side y"][-1]
            ):
                return "win"
            else:
                return "reset"
        # reset condition
        # left side
        i1 = np.where(self.trackmap["left side x"] == state0[0])
        i2 = np.where(self.trackmap["left side x"] == state1[0])
        point1 = [-1, -1]
        if state0[1] < state1[1]:
            ymin = state0[1]
            ymax = state1[1]
        else:
            ymin = state1[1]
            ymax = state0[1]
        if (
            (self.trackmap["left side y"][i1] >= ymin)
            & (self.trackmap["left side y"][i1] <= ymax)
        ).any():
            point1[0] = 1
        if (
            (self.trackmap["left side y"][i2] >= ymin)
            & (self.trackmap["left side y"][i2] <= ymax)
        ).any():
            point1[1] = 1
        if point1[0] > 0 and point1[1] > 0:
            return "reset"
        j1 = np.where(self.trackmap["left side y"] == state0[1])
        j2 = np.where(self.trackmap["left side y"] == state1[1])
        point2 = [-1, -1]
        if state0[0] < state1[0]:
            xmin = state0[0]
            xmax = state1[0]
        else:
            xmin = state1[0]
            xmax = state0[0]
        if (
            (self.trackmap["left side x"][j1] >= xmin)
            & (self.trackmap["left side x"][j1] <= xmax)
        ).any():
            point2[0] = 1
        if (
            (self.trackmap["left side x"][j2] >= xmin)
            & (self.trackmap["left side x"][j2] <= xmax)
        ).any():
            point2[1] = 1
        if point2[0] > 0 and point2[1] > 0:
            return "reset"
        elif point1[0] > 0 and point2[0] > 0:
            return "reset"
        elif point1[1] > 0 and point2[1] > 0:
            return "reset"
        # right side
        i1 = np.where(self.trackmap["right side x"] == state0[0])
        i2 = np.where(self.trackmap["right side x"] == state1[0])
        point1 = [-1, -1]
        if state0[1] < state1[1]:
            ymin = state0[1]
            ymax = state1[1]
        else:
            ymin = state1[1]
            ymax = state0[1]
        if (
            (self.trackmap["right side y"][i1] >= ymin)
            & (self.trackmap["right side y"][i1] <= ymax)
        ).any():
            point1[0] = 1
        if (
            (self.trackmap["right side y"][i2] >= ymin)
            & (self.trackmap["right side y"][i2] <= ymax)
        ).any():
            point1[1] = 1
        if point1[0] > 0 and point1[1] > 0:
            return "reset"
        j1 = np.where(self.trackmap["right side y"] == state0[1])
        j2 = np.where(self.trackmap["right side y"] == state1[1])
        point2 = [-1, -1]
        if state0[0] < state1[0]:
            xmin = state0[0]
            xmax = state1[0]
        else:
            xmin = state1[0]
            xmax = state0[0]
        if (
            (self.trackmap["right side x"][j1] >= xmin)
            & (self.trackmap["right side x"][j1] <= xmax)
        ).any():
            point2[0] = 1
        if (
            (self.trackmap["right side x"][j2] >= xmin)
            & (self.trackmap["right side x"][j2] <= xmax)
        ).any():
            point2[1] = 1
        if point2[0] > 0 and point2[1] > 0:
            return "reset"
        elif point1[0] > 0 and point2[0] > 0:
            return "reset"
        elif point1[1] > 0 and point2[1] > 0:
            return "reset"

        return "continue"

    def evaluate_time(self, data=[]):
        _, ax = plt.subplots()
        if len(data) > 0:
            for i in range(self.N):
                ax.fill_between(
                    range(self.epi),
                    np.percentile(data[i, :, :], 75, axis=1),
                    np.min(data[i, :, :], axis=1),
                    alpha=0.5,
                    linewidth=0,
                )
                ax.plot(
                    np.percentile(data[i, :, :], 50, axis=1),
                    "-",
                    label=self.agent_names[i]["name"],
                )
        else:
            for i in range(self.N):
                ax.fill_between(
                    range(self.epi),
                    np.percentile(self.t[i, :, :], 75, axis=1),
                    np.min(self.t[i, :, :], axis=1),
                    alpha=0.5,
                    linewidth=0,
                )
                ax.plot(
                    np.percentile(self.t[i, :, :], 50, axis=1),
                    "-",
                    label=self.agent_names[i]["name"],
                )
        ax.legend()
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Time")
        ax.set_xlim(0, self.epi)
        # plt.grid("on")
        plt.savefig("./image/per-episode times.png")
        plt.show()
        return self.t

    def evaluate_resetfreq(self, data=[]):
        _, ax = plt.subplots()
        if len(data) > 0:
            for i in range(self.N):
                ax.fill_between(
                    range(self.epi),
                    np.percentile(data[i, :, :], 75, axis=1),
                    np.min(data[i, :, :], axis=1),
                    alpha=0.5,
                    linewidth=0,
                )
                ax.plot(
                    np.percentile(data[i, :, :], 50, axis=1),
                    "-",
                    label=self.agent_names[i]["name"],
                )
        else:
            for i in range(self.N):
                ax.fill_between(
                    range(self.epi),
                    np.percentile(self.reset_freq[i, :, :], 75, axis=1),
                    np.min(self.reset_freq[i, :, :], axis=1),
                    alpha=0.5,
                    linewidth=0,
                )
                ax.plot(
                    np.percentile(self.reset_freq[i, :, :], 50, axis=1),
                    "-",
                    label=self.agent_names[i]["name"],
                )
        ax.legend()
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reset times")
        ax.set_xlim(0, self.epi)
        # plt.grid("on")
        plt.savefig("./image/per-episode reset times.png")
        plt.show()
        return self.reset_freq
