import numpy as np
import copy


class Agent(object):
    """
    name: type of algorithm
    eps: epsilon
    alpha: learning rate
    gamma: gamma
    alphax: The rate at which alpha decays when win
    s: variables of the value function
    s_max: the upper bound of s
    g_win: additive gain of reward when win
    g_reset: additive gain of reward when reset
    if_record: whether record variables of v and pi
    action: action selected per period
    K: number of actions for each direction
    v: estimated value function
    dv: gradient of the value function
    w: parameters of the value function, initialized in random
    theta: parameters of policy, initialized in random
    n: number of steps
    piv: variables of pi

    selection(): selecting an action
    V(): value function
    update(): updating parameters
    get_w_theta(): returning parameters w, theta
    get_s_piv(): returning variables of v and pi
    """

    def __init__(
        self,
        name,
        trackmap,
        eps=0.1,
        alpha=np.array([0.01, 0.001]),
        gamma=0.1,
        alphax=0.9,
        s_max=16,
        g_win=2,
        g_reset=-20,
        if_record=False,
    ):
        self.__name = name
        self.trackmap = trackmap
        self.__eps = eps
        self.__alpha = alpha
        self.__gamma = gamma
        self.__alphax = alphax
        self.__s_max = s_max
        self.__g_win = g_win
        self.__g_reset = g_reset

        self.__action = [0, 0]
        self.__K = [3, 3]
        self.__v = []
        self.__dv = []
        self.__dpi = []
        self.__r = []
        self.__w = np.random.normal(4, 1, size=6) * 0.1
        self.__theta = np.random.normal(4, 1, size=3)
        self.__theta[1:3] *= 0.1
        self.__n = 1
        self.__if_record = if_record
        self.__s = {}
        self.__piv = {}

        self.__w[self.__w < 0] = 1
        self.__theta[self.__theta < 0] = 1

        # self.__w = np.array(
        #     [0.00048192, 0.04342782, 0.47394036, 0.47556801, 0.41361428, 0.41390277]
        # )
        # self.__theta = np.array([3.13245097, 0.26531583, 0.51152292])

    # def __del__(self):
    #     print("agent will be deleted!")

    def selection(self, state):
        if self.__name == "policy-gradient (decay-epsilon greedy)":
            if np.random.random() < self.__eps / np.sqrt(self.__n):
                while 1:
                    self.__action = np.random.randint(self.__K[0], size=2) - 1
                    state1 = np.zeros(2)
                    state1[0] = state["speed"][0] + (self.__action[0])
                    state1[1] = state["speed"][1] + (self.__action[1])
                    if not (
                        (state1 < 0).any() or (state1 > 5).any() or (state1 == 0).all()
                    ):
                        break
            else:
                self.__action = [0, 0]
                pi = -np.inf
                for i in range(self.__K[0]):
                    for j in range(self.__K[1]):
                        state1 = copy.deepcopy(state)
                        state1["speed"][0] = state["speed"][0] + (i - 1)
                        state1["speed"][1] = state["speed"][1] + (j - 1)
                        if (
                            (state1["speed"] < 0).any()
                            or (state1["speed"] > 5).any()
                            or (state1["speed"] == 0).all()
                        ):
                            continue
                        state1["position"][0] = (
                            state["position"][0] + state1["speed"][0]
                        )
                        state1["position"][1] = (
                            state["position"][1] + state1["speed"][1]
                        )
                        tmp, _ = self.V(copy.deepcopy(state1))

                        prefer = np.dot(self.__theta, tmp)

                        if self.__if_record:
                            if str(state1["position"]) in self.__piv:
                                self.__piv[str(state1["position"])].append(tmp)
                            else:
                                self.__piv[str(state1["position"])] = [tmp]

                        del state1

                        if prefer > pi:
                            self.__action = [i - 1, j - 1]
                            pi = prefer
        elif self.__name == "policy-gradient (softmax)":
            self.__action = [0, 0]
            pi, actions = [], []
            for i in range(self.__K[0]):
                for j in range(self.__K[1]):
                    state1 = copy.deepcopy(state)
                    state1["speed"][0] = state["speed"][0] + (i - 1)
                    state1["speed"][1] = state["speed"][1] + (j - 1)
                    if (
                        (state1["speed"] < 0).any()
                        or (state1["speed"] > 5).any()
                        or (state1["speed"] == 0).all()
                    ):
                        continue
                    state1["position"][0] = state["position"][0] + state1["speed"][0]
                    state1["position"][1] = state["position"][1] + state1["speed"][1]
                    tmp, _ = self.V(copy.deepcopy(state1))

                    if self.__if_record:
                        if str(state1["position"]) in self.__piv:
                            self.__piv[str(state1["position"])].append(tmp)
                        else:
                            self.__piv[str(state1["position"])] = [tmp]

                    del state1
                    prefer = np.dot(self.__theta, tmp)
                    pi.append(prefer)
                    actions.append([i - 1, j - 1])

            pi = np.array(pi)
            actions = np.array(actions)
            pi *= (self.__n + 10) ^ 2  # f(n)
            pi -= np.max(pi)
            p = np.exp(pi) / np.sum(np.exp(pi))
            self.__action = actions[np.random.multinomial(n=1, pvals=p) == 1][0]

        del state
        return self.__action

    def V(self, state):
        if "policy-gradient" in self.__name:
            position = state["position"]
            speed = state["speed"]
            s = np.zeros(6)
            j1 = np.where(self.trackmap["left side y"] == position[1])
            j2 = np.where(self.trackmap["right side y"] == position[1])
            i1 = np.where(self.trackmap["left side x"] == position[0])
            i2 = np.where(self.trackmap["right side x"] == position[0])
            l1, r1, u1, d1 = -1, -1, -1, -1
            # 左右距离边界
            if (j1[0] + 1).any():
                l1 = position[0] - max(self.trackmap["left side x"][j1]) - 1
                s[0] = l1
            if (j2[0] + 1).any():
                r1 = min(self.trackmap["right side x"][j2]) - position[0] - 1
                if r1 < l1:
                    s[0] = r1
            # 上下距离边界
            if (i1[0] + 1).any():
                u1 = min(self.trackmap["left side y"][i1]) - position[1] - 1
                s[1] = u1
            if (i2[0] + 1).any():
                d1 = position[1] - max(self.trackmap["right side y"][i2]) - 1
                if d1 < u1:
                    s[1] = d1
            # 横向距离终点
            s[2] = -np.log(
                abs(self.trackmap["left side x"][-1] - position[0])
                + (speed[0] * 0.5 + 1)
            )
            # 纵向距离终点
            s[3] = -np.log(
                abs(self.trackmap["right side y"][-1] - position[1])
                + (speed[1] * 0.5 + 1)
            )
            s[4] = s[0] * np.exp(s[2])
            s[5] = s[1] * np.exp(s[3])

            s[s > self.__s_max] = self.__s_max
            s[s < -self.__s_max] = -self.__s_max

            if self.__if_record:
                if str(position) in self.__s:
                    self.__s[str(position)].append(s)
                else:
                    self.__s[str(position)] = [s]

            v = np.dot(self.__w, s)
            del state
            return (
                np.array(
                    [
                        v,
                        np.log(max(s[0], 0) * speed[0] + 1),
                        np.log(max(s[1], 0) * speed[1] + 1),
                    ]
                ),
                s,
            )

    def update(self, state0, r, state1, referee):
        if "policy-gradient" in self.__name:
            # print("w: ", self.__w, "theta: ", self.__theta)
            if referee == "reset":
                r += self.__g_reset
            elif referee == "win":
                r += self.__g_win
            tmp1, tmp2 = self.V(state0)
            self.__v.append(tmp1[0])
            self.__dv.append(tmp2)
            self.__dpi.append(tmp1 / np.dot(self.__theta, tmp1))
            self.__r.append(r)
            self.__n += 1

            if referee in ("reset", "win"):
                v1, _ = self.V(state1)
                delta = r + self.__gamma * v1[0] - self.__v[-1]
                t = len(self.__v) - 1
                self.__v.append(v1[0])
                while t >= 0:
                    delta = self.__r[t] + self.__gamma * self.__v[t + 1] - self.__v[t]
                    para = 1
                    while 1:
                        tmp = (
                            self.__theta
                            + para
                            * self.__alpha[0]
                            * np.power(self.__gamma, t)
                            * delta
                            * self.__dpi[t]
                        )
                        if tmp[0] >= 0:
                            self.__theta = tmp
                            break
                        else:
                            para /= 2
                    para = 1
                    while 1:
                        tmp = self.__w + para * self.__alpha[1] * delta * self.__dv[t]
                        if (tmp[:4] >= 0).all():
                            self.__w = tmp
                            break
                        else:
                            para /= 2
                    t = t - 1
                if referee == "win":
                    self.__alpha = self.__alpha * self.__alphax
                self.__v = []
                self.__dv = []
                self.__dpi = []
                self.__r = []

    def get_w_theta(self):
        return self.__w.copy(), self.__theta.copy()

    def get_s_piv(self):
        return self.__s, self.__piv
