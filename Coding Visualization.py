import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim


def lock_seeds(seed=42):
    """Locks all random generators to ensure identical simulation runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────
# 1. CONFIG & ENVIRONMENT
# ─────────────────────────────────────────────
ENV_CONFIG = {
    "N": 1000.0,
    "beta": 0.12,
    "sigma": 1.0,
    "gamma": 1 / 27,
    "mu": 0.009,
    "max_days": 365,
    "zeta_values": [0.00, 0.25, 0.50, 0.65],
    "economy_outputs": [1.0, 0.75, 0.50, 0.35],
    "state_size": 6,
    "action_size": 4,
}


class RewardCalculator:
    def __init__(self, agent_type="balanced"):
        self.r = 12 if agent_type == "balanced" else 9
        self.s = 5.0

    def calculate_reward(self, E_econ, A, D):
        return (E_econ * np.exp(-self.r * A)) - (self.s * D)


class EpidemicEnv:
    def __init__(self, agent_type="balanced"):
        self.rc = RewardCalculator(agent_type)

    def reset(self):
        self.S, self.E, self.I, self.R, self.D, self.E_econ = (
            990.0,
            10.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.day = 0
        self.done = False
        return np.array([self.S, self.E, self.I, self.R, self.D, self.E_econ])

    def step(self, act):
        z = ENV_CONFIG["zeta_values"][act]
        ec = ENV_CONFIG["economy_outputs"][act]
        nE = ENV_CONFIG["beta"] * (1 - z) * self.S * self.I / ENV_CONFIG["N"]
        nI = ENV_CONFIG["sigma"] * self.E
        nR = ENV_CONFIG["gamma"] * self.I
        nD = ENV_CONFIG["mu"] * self.I
        self.S -= nE
        self.E += nE - nI
        self.I += nI - nR - nD
        self.R += nR
        self.D += nD
        self.E_econ += ec
        self.day += 1
        rew = self.rc.calculate_reward(
            ec, self.I / ENV_CONFIG["N"], nD / ENV_CONFIG["N"]
        )
        if self.day >= ENV_CONFIG["max_days"] or (self.I < 0.1 and self.E < 0.1):
            self.done = True
        return (
            np.array([self.S, self.E, self.I, self.R, self.D, self.E_econ]),
            rew,
            self.done,
            nD,
        )


# ─────────────────────────────────────────────
# 2. SHARED UTILITIES
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, cap=10000):
        self.buf = deque(maxlen=cap)

    def push(self, *a):
        self.buf.append(a)

    def __len__(self):
        return len(self.buf)

    def sample(self, n):
        b = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*b)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(d),
        )


# ─────────────────────────────────────────────
# 3. D3QN  (Dueling Double DQN)
# ─────────────────────────────────────────────
class DuelingQNet(nn.Module):
    def __init__(self, ss, ac):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(ss, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.val = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, ac))

    def forward(self, x):
        f = self.feat(x)
        v, a = self.val(f), self.adv(f)
        return v + (a - a.mean(1, keepdim=True))


class D3QNAgent:
    def __init__(self, ss, ac, lr, gamma, expl="epsilon-greedy"):
        self.ac = ac
        self.gamma = gamma
        self.expl = expl
        self.eps = 1.0
        self.temp = 1.0
        self.steps = 0
        self.qnet = DuelingQNet(ss, ac)
        self.tnet = DuelingQNet(ss, ac)
        self.tnet.load_state_dict(self.qnet.state_dict())
        self.opt = optim.Adam(self.qnet.parameters(), lr=lr)
        self.mem = ReplayBuffer(10000)

    def select_action(self, state):
        qt = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.qnet(qt).squeeze()
        if self.expl == "epsilon-greedy":
            return (
                random.randint(0, self.ac - 1)
                if random.random() < self.eps
                else q.argmax().item()
            )
        return np.random.choice(self.ac, p=torch.softmax(q / self.temp, 0).numpy())

    def step_train(self, s, a, r, ns, done, **kw):
        self.mem.push(s, a, r, ns, done)
        if len(self.mem) < 64:
            return
        S, A, R, NS, D = self.mem.sample(64)
        with torch.no_grad():
            na = self.qnet(NS).argmax(1, keepdim=True)
            nq = self.tnet(NS).gather(1, na).squeeze()
            tgt = R + (1 - D) * self.gamma * nq
        cq = self.qnet(S).gather(1, A.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(cq, tgt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        if self.steps % 10 == 0:
            self.tnet.load_state_dict(self.qnet.state_dict())

    def end_episode(self):
        pass

    def reset_for_eval(self):
        pass

    def update_exploration(self):
        self.eps = max(self.eps * 0.995, 0.01)
        self.temp = max(self.temp * 0.995, 0.01)


# ─────────────────────────────────────────────
# 4. CPO  (Constrained Policy Optimization)
#    Core idea: PPO-style policy gradient with a
#    Lagrangian multiplier penalising death-cost
#    violations. λ rises when deaths exceed limit.
# ─────────────────────────────────────────────
class _MLP(nn.Module):
    def __init__(self, dims, out_act=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [nn.ReLU()]
        if out_act:
            layers += [out_act]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CPOAgent:
    def __init__(self, ss, ac, lr=1e-3, gamma=0.99, cost_limit=0.002):
        self.ac = ac
        self.gamma = gamma
        self.cost_limit = cost_limit
        self.policy = _MLP([ss, 128, 128, ac], nn.Softmax(dim=-1))
        self.vnet = _MLP([ss, 128, 64, 1])
        self.cvnet = _MLP([ss, 128, 64, 1])  # cost value baseline
        self.pol_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.val_opt = optim.Adam(
            list(self.vnet.parameters()) + list(self.cvnet.parameters()), lr=lr
        )
        # Learnable Lagrange multiplier (clamped ≥ 0)
        self.lam = torch.tensor(0.5, requires_grad=True)
        self.lam_opt = optim.Adam([self.lam], lr=0.02)
        self.traj = []
        self.eps = 1.0

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.ac - 1)
        with torch.no_grad():
            p = self.policy(torch.FloatTensor(state).unsqueeze(0)).squeeze()
        if self.eps == 0.0:
            return torch.argmax(p).item()
        return torch.multinomial(p, 1).item()

    def step_train(self, s, a, r, ns, done, cost=0.0):
        self.traj.append((s, a, r, cost, ns, float(done)))

    def end_episode(self):
        if len(self.traj) < 2:
            self.traj = []
            return
        S, A, R, C, NS, D = map(np.array, zip(*self.traj))
        s = torch.FloatTensor(S)
        a = torch.LongTensor(A.astype(int))
        r = torch.FloatTensor(R)
        c = torch.FloatTensor(C)
        ns = torch.FloatTensor(NS)
        d = torch.FloatTensor(D)
        with torch.no_grad():
            v = self.vnet(s).squeeze()
            nv = self.vnet(ns).squeeze()
            cv = self.cvnet(s).squeeze()
            ncv = self.cvnet(ns).squeeze()
            adv = r + self.gamma * nv * (1 - d) - v
            cadv = c + self.gamma * ncv * (1 - d) - cv
            vt = (r + self.gamma * nv * (1 - d)).detach()
            cvt = (c + self.gamma * ncv * (1 - d)).detach()
        # ── Policy update (Lagrangian objective) ──
        probs = self.policy(s)
        lp = torch.log(probs.gather(1, a.unsqueeze(1)).squeeze() + 1e-8)
        lam = self.lam.clamp(min=0.0)
        pol_loss = -(lp * (adv - lam.detach() * cadv)).mean()
        self.pol_opt.zero_grad()
        pol_loss.backward()
        self.pol_opt.step()
        # ── Value baselines ──
        vl = nn.MSELoss()(self.vnet(s).squeeze(), vt) + nn.MSELoss()(
            self.cvnet(s).squeeze(), cvt
        )
        self.val_opt.zero_grad()
        vl.backward()
        self.val_opt.step()
        # ── Dual ascent: λ rises when mean cost > limit ──
        lag_loss = -self.lam * (c.mean() - self.cost_limit)
        self.lam_opt.zero_grad()
        lag_loss.backward()
        self.lam_opt.step()
        self.traj = []

    def reset_for_eval(self):
        pass

    def update_exploration(self):
        self.eps = max(self.eps * 0.995, 0.01)


# ─────────────────────────────────────────────
# 5. DECISION TRANSFORMER
#    Core idea: treat (RTG, state, action) triples
#    as a sequence; train a causal Transformer to
#    predict the next action given a desired return.
# ─────────────────────────────────────────────
class _DT(nn.Module):
    def __init__(self, ss, ac, dm=64, nh=4, nl=2, ctx=20):
        super().__init__()
        self.ctx = ctx
        self.dm = dm
        self.ac = ac
        self.se = nn.Linear(ss, dm)
        self.ae = nn.Embedding(ac, dm)
        self.re = nn.Linear(1, dm)
        self.pe = nn.Embedding(3 * ctx, dm)
        enc = nn.TransformerEncoderLayer(
            dm, nh, dim_feedforward=128, batch_first=True, dropout=0.0
        )
        self.tf = nn.TransformerEncoder(enc, nl)
        self.hd = nn.Linear(dm, ac)

    def forward(self, rtg, st, at):
        B, T = st.shape[:2]
        # Ensure rtg is always [B, T, 1] regardless of how it was passed in
        if rtg.dim() == 2:
            rtg = rtg.unsqueeze(-1)  # [B,T] -> [B,T,1]
        r_emb = self.re(rtg)  # [B,T,dm]
        s_emb = self.se(st)  # [B,T,dm]
        a_emb = self.ae(at)  # [B,T,dm]
        # Interleave as (rtg, state, action) triples along the time axis
        seq = torch.cat(
            [
                r_emb.unsqueeze(2),  # [B,T,1,dm]
                s_emb.unsqueeze(2),  # [B,T,1,dm]
                a_emb.unsqueeze(2),  # [B,T,1,dm]
            ],
            dim=2,
        ).reshape(B, 3 * T, self.dm)  # [B,3T,dm]
        seq = seq + self.pe(torch.arange(3 * T, device=st.device))
        mask = nn.Transformer.generate_square_subsequent_mask(3 * T).to(st.device)
        out = self.tf(seq, mask=mask, is_causal=True)
        return self.hd(out[:, 1::3, :])  # state-token predictions [B,T,ac]


class DTAgent:
    def __init__(self, ss, ac, lr=1e-3, ctx=20, target_rtg=60.0):
        self.ss = ss
        self.ac = ac
        self.ctx = ctx
        self.trtg = target_rtg
        self.model = _DT(ss, ac, ctx=ctx)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.dataset = []
        self.cur = {"s": [], "a": [], "r": []}
        self.trained = False
        self.eps = 1.0
        self._reset_ctx()

    def _reset_ctx(self):
        self.ctx_s = []
        self.ctx_a = []
        self.ctx_r = []
        self.cur_rtg = self.trtg

    def select_action(self, state):
        if not self.trained or random.random() < self.eps:
            return random.randint(0, self.ac - 1)
        self.ctx_s.append(state)
        self.ctx_r.append(self.cur_rtg)
        n = len(self.ctx_s)

        T = min(n, self.ctx)
        s = np.array(self.ctx_s[-T:], dtype=np.float32)
        rtg = np.array(self.ctx_r[-T:], dtype=np.float32)
        past_a = (
            list(self.ctx_a[-(T - 1) :]) if self.ctx_a else []
        )  # at most T-1 real actions
        dummy_a = (
            [0] * (T - len(past_a)) + past_a + [0]
        )  # left-pad + dummy for current step → always length T
        a = np.array(dummy_a[:T], dtype=np.int64)

        assert len(s) == len(a) == len(rtg) == T, (
            f"Length mismatch: s={len(s)}, a={len(a)}, rtg={len(rtg)}, T={T}"
        )
        pad = self.ctx - T
        if pad > 0:
            s = np.vstack([np.zeros((pad, self.ss), dtype=np.float32), s])
            a = np.concatenate([np.zeros(pad, dtype=np.int64), a])
            rtg = np.concatenate([np.zeros(pad, dtype=np.float32), rtg])
        rtg = rtg / (np.abs(rtg).max() + 1e-8)
        st = torch.FloatTensor(s).unsqueeze(0)  # [1, ctx, ss]
        at = torch.LongTensor(a).unsqueeze(0)  # [1, ctx]
        rt = torch.FloatTensor(rtg).unsqueeze(0).unsqueeze(-1)  # [1, ctx, 1]
        self.model.eval()
        with torch.no_grad():
            logits = self.model(rt, st, at)  # [1, ctx, ac]
        return logits[0, T - 1].argmax().item()

    def step_train(self, s, a, r, ns, done, **kw):
        self.cur["s"].append(s)
        self.cur["a"].append(a)
        self.cur["r"].append(r)
        self.ctx_a.append(a)
        self.cur_rtg -= r

    def end_episode(self):
        if self.cur["s"]:
            self.dataset.append(dict(self.cur))
        self.cur = {"s": [], "a": [], "r": []}
        self._reset_ctx()
        if len(self.dataset) >= 5:
            self._fit(150)

    def _fit(self, steps):
        self.model.train()
        for _ in range(steps):
            traj = random.choice(self.dataset)
            T = len(traj["s"])
            if T < 2:
                continue
            st = random.randint(0, max(0, T - self.ctx))
            en = min(st + self.ctx, T)
            L = en - st
            s = np.array(traj["s"][st:en])
            a = np.array(traj["a"][st:en])
            rw = np.array(traj["r"][st:en])
            rtg = np.zeros(L)
            cum = 0
            for i in range(L - 1, -1, -1):
                cum += rw[i]
                rtg[i] = cum
            rtg /= abs(rtg).max() + 1e-8
            pad = self.ctx - L
            if pad:
                s = np.vstack([s, np.zeros((pad, self.ss))])
                a = np.concatenate([a, np.zeros(pad, dtype=int)])
                rtg = np.concatenate([rtg, np.zeros(pad)])
            sl = torch.FloatTensor(s).unsqueeze(0)
            al = torch.LongTensor(a).unsqueeze(0)
            rl = torch.FloatTensor(rtg).unsqueeze(0).unsqueeze(-1)
            logits = self.model(rl, sl, al)
            loss = nn.CrossEntropyLoss()(logits[0, :L], al[0, :L])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.trained = True

    def reset_for_eval(self):
        self._reset_ctx()

    def update_exploration(self):
        self.eps = max(self.eps * 0.99, 0.05)


# 6. DREAMERV3  (Simplified)
class _WorldModel(nn.Module):
    def __init__(self, ss, ac, lat=32, hid=64):
        super().__init__()
        self.lat = lat
        self.hid = hid
        self.enc = nn.Sequential(nn.Linear(ss, 64), nn.ReLU(), nn.Linear(64, lat * 2))
        self.gru = nn.GRUCell(lat + ac, hid)
        self.prior = nn.Sequential(
            nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, lat * 2)
        )
        self.dec = nn.Sequential(nn.Linear(lat + hid, 64), nn.ReLU(), nn.Linear(64, ss))
        self.rew = nn.Sequential(nn.Linear(lat + hid, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, x):
        o = self.enc(x)
        return o.chunk(2, -1)

    def reparam(self, m, lv):
        return m + torch.randn_like(m) * torch.exp(0.5 * lv)

    def imagine(self, z, h, a):
        h2 = self.gru(torch.cat([z, a], -1), h)
        pm, plv = self.prior(h2).chunk(2, -1)
        return self.reparam(pm, plv), h2

    def decode(self, z, h):
        return self.dec(torch.cat([z, h], -1))

    def pred_rew(self, z, h):
        return self.rew(torch.cat([z, h], -1)).squeeze(-1)


def _safe_multinomial(p):
    """Sample from a probability tensor, falling back to uniform if nan/inf present."""
    p = p.float()
    if not torch.isfinite(p).all() or (p < 0).any():
        return torch.randint(0, p.shape[-1], (p.shape[0],))
    # Re-normalise to guard against floating-point drift
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    p = p.clamp(min=0.0)
    return torch.multinomial(p, 1).squeeze(1)


class DreamerAgent:
    def __init__(self, ss, ac, lr=1e-3, gamma=0.99):
        self.ss = ss
        self.ac = ac
        self.gamma = gamma
        self.lat = 32
        self.hid = 64
        self.wm = _WorldModel(ss, ac, self.lat, self.hid)
        # Use no built-in Softmax — we apply it manually with clamping at sample time
        self.actor = _MLP([self.lat + self.hid, 64, ac])
        self.critic = _MLP([self.lat + self.hid, 64, 1])
        self.wm_opt = optim.Adam(self.wm.parameters(), lr=lr)
        self.ac_opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.mem = ReplayBuffer(10000)
        self.h = torch.zeros(1, self.hid)
        self.eps = 1.0

    def _feat(self, z, h):
        return torch.cat([z, h], -1)

    def _actor_probs(self, feat):
        """Numerically stable softmax: clamp logits before softmax, re-normalise after."""
        logits = self.actor(feat).clamp(-10, 10)
        p = torch.softmax(logits, dim=-1)
        p = p.clamp(min=1e-6)
        return p / p.sum(dim=-1, keepdim=True)

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.ac - 1)
        obs = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            m, lv = self.wm.encode(obs)
            lv = lv.clamp(-4, 4)
            if self.eps == 0.0:
                z = m
            else:
                z = self.wm.reparam(m, lv)
            p = self._actor_probs(self._feat(z, self.h))

        if self.eps == 0.0:
            return torch.argmax(p).item()
        return _safe_multinomial(p).item()

    def step_train(self, s, a, r, ns, done, **kw):
        self.mem.push(s, a, r, ns, float(done))
        obs = torch.FloatTensor(s).unsqueeze(0)
        aoh = torch.zeros(1, self.ac)
        aoh[0, a] = 1.0
        with torch.no_grad():
            m, lv = self.wm.encode(obs)
            lv = lv.clamp(-4, 4)
            z = self.wm.reparam(m, lv)
            _, self.h = self.wm.imagine(z, self.h, aoh)
        if len(self.mem) >= 64:
            self._train()

    def _train(self):
        B = 32
        S, A, R, NS, D = self.mem.sample(B)
        # ── World model ──
        m, lv = self.wm.encode(S)
        lv = lv.clamp(-4, 4)  # clamp before exp()
        z = self.wm.reparam(m, lv)
        h0 = torch.zeros(B, self.hid)
        aoh = torch.zeros(B, self.ac)
        aoh.scatter_(1, A.unsqueeze(1), 1.0)
        zn, hn = self.wm.imagine(z, h0, aoh)
        kl = (-0.5 * (1 + lv - m.pow(2) - lv.exp())).sum(1).mean()
        kl = kl.clamp(max=100.0)  # cap KL so it can't explode
        wm_loss = (
            nn.MSELoss()(self.wm.decode(z, h0), S)
            + nn.MSELoss()(self.wm.pred_rew(zn, hn), R)
            + 0.1 * kl
        )
        self.wm_opt.zero_grad()
        wm_loss.backward()
        self.wm_opt.step()
        # ── Imagination rollout (H=5 steps) ──
        H = 5
        zs, hs = zn.detach()[:16], hn.detach()[:16]
        ir, iv = [], []
        for _ in range(H):
            p = self._actor_probs(self._feat(zs, hs))
            a = _safe_multinomial(p)
            ao = torch.zeros(16, self.ac)
            ao.scatter_(1, a.unsqueeze(1), 1.0)
            with torch.no_grad():
                zs, hs = self.wm.imagine(zs.detach(), hs.detach(), ao)
            ir.append(self.wm.pred_rew(zs, hs))
            iv.append(self.critic(self._feat(zs, hs)).squeeze())
        # ── Lambda returns (γ-discounted) ──
        Rv = iv[-1].detach()
        rets = []
        for t in range(H - 1, -1, -1):
            Rv = ir[t] + self.gamma * Rv
            rets.insert(0, Rv)
        rt = torch.stack(rets)
        vt = torch.stack(iv)
        adv = rt - vt.detach()
        ac_loss = -adv.mean() + nn.MSELoss()(vt, rt.detach())
        self.ac_opt.zero_grad()
        ac_loss.backward()
        self.ac_opt.step()

    def end_episode(self):
        self.h = torch.zeros(1, self.hid)

    def reset_for_eval(self):
        self.h = torch.zeros(1, self.hid)

    def update_exploration(self):
        self.eps = max(self.eps * 0.995, 0.01)


# ─────────────────────────────────────────────
# 7. MUZERO  (Simplified)
#    Core idea: learn representation h(obs)→s,
#    dynamics g(s,a)→(s',r), prediction f(s)→(π,v).
#    Plan at decision time using MCTS over the
#    learned model (no real env rollouts needed).
# ─────────────────────────────────────────────
class _MuNet(nn.Module):
    def __init__(self, ss, ac, hs=64):
        super().__init__()
        self.ac = ac
        self.h = nn.Sequential(
            nn.Linear(ss, 64), nn.ReLU(), nn.Linear(64, hs), nn.Tanh()
        )
        self.gs = nn.Sequential(
            nn.Linear(hs + ac, 64), nn.ReLU(), nn.Linear(64, hs), nn.Tanh()
        )
        self.gr = nn.Sequential(nn.Linear(hs + ac, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fp = nn.Sequential(nn.Linear(hs, 64), nn.ReLU(), nn.Linear(64, ac))
        self.fv = nn.Sequential(nn.Linear(hs, 64), nn.ReLU(), nn.Linear(64, 1))

    def represent(self, o):
        return self.h(o)

    def dynamics(self, s, a):
        inp = torch.cat([s, a], -1)
        return self.gs(inp), self.gr(inp).squeeze(-1)

    def predict(self, s):
        return self.fp(s), self.fv(s).squeeze(-1)


class MuZeroAgent:
    def __init__(self, ss, ac, lr=1e-3, gamma=0.99, n_sim=10):
        self.ss = ss
        self.ac = ac
        self.gamma = gamma
        self.nsim = n_sim
        self.hs = 64
        self.net = _MuNet(ss, ac, self.hs)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.mem = ReplayBuffer(10000)
        self.eps = 1.0
        self.steps = 0

    def _mcts(self, hidden):
        """UCB1 tree search over n_sim simulations using learned model."""
        with torch.no_grad():
            pl, rv = self.net.predict(hidden)
            prior = torch.softmax(pl, dim=-1).squeeze().numpy()
        visits = np.zeros(self.ac)
        vals = np.zeros(self.ac)
        for _ in range(self.nsim):
            # UCB selection
            ucb = vals / (visits + 1e-8) + prior * np.sqrt(visits.sum() + 1) / (
                visits + 1
            )
            a = int(np.argmax(ucb))
            aoh = torch.zeros(1, self.ac)
            aoh[0, a] = 1.0
            with torch.no_grad():
                nh, pr = self.net.dynamics(hidden, aoh)
                _, v = self.net.predict(nh)
            visits[a] += 1
            vals[a] += pr.item() + self.gamma * v.item()
        return int(np.argmax(visits))

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.ac - 1)
        h = self.net.represent(torch.FloatTensor(state).unsqueeze(0))
        return self._mcts(h)

    def step_train(self, s, a, r, ns, done, **kw):
        self.mem.push(s, a, r, ns, float(done))
        if len(self.mem) < 64:
            return
        S, A, R, NS, D = self.mem.sample(64)
        hid = self.net.represent(S)
        pl, v = self.net.predict(hid)
        aoh = torch.zeros(64, self.ac)
        aoh.scatter_(1, A.unsqueeze(1), 1.0)
        nh, pr = self.net.dynamics(hid, aoh)
        with torch.no_grad():
            _, nv = self.net.predict(self.net.represent(NS))
            vt = R + self.gamma * nv * (1 - D)
        loss = nn.MSELoss()(v, vt) + nn.MSELoss()(pr, R) + nn.CrossEntropyLoss()(pl, A)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1

    def end_episode(self):
        pass

    def reset_for_eval(self):
        pass

    def update_exploration(self):
        self.eps = max(self.eps * 0.995, 0.01)


# ─────────────────────────────────────────────
# 8. AGENT FACTORY
# ─────────────────────────────────────────────
ALGO_INFO = {
    "D3QN": "**Dueling Double DQN** — value-based, experience replay, separate advantage/value streams.",
    "CPO": "**Constrained Policy Optimization** — PPO backbone with a Lagrangian multiplier that "
    "rises when daily deaths exceed a safety threshold. Balances reward *and* safety.",
    "Decision Transformer": "**Decision Transformer** — offline sequence model. First collects experience, then trains "
    "a causal Transformer to predict actions conditioneD on a desired Return-To-Go.",
    "DreamerV3": "**DreamerV3-lite** — GRU world model learns environment dynamics in a 32-dim latent space. "
    "Actor-critic is trained purely on *imagined* rollouts — zero extra real-env steps.",
    "MuZero": "**MuZero-lite** — learns representation h(obs), dynamics g(s,a)→(s',r), and prediction "
    "f(s)→(π,v) networks. Plans at decision time using Monte-Carlo Tree Search (MCTS) "
    "over the learned model.",
}


def make_agent(algo, ss, ac, lr, gamma, agent_type, expl):
    if algo == "D3QN":
        return D3QNAgent(ss, ac, lr, gamma, expl)
    if algo == "CPO":
        return CPOAgent(ss, ac, lr, gamma)
    if algo == "Decision Transformer":
        return DTAgent(ss, ac, lr)
    if algo == "DreamerV3":
        return DreamerAgent(ss, ac, lr, gamma)
    if algo == "MuZero":
        return MuZeroAgent(ss, ac, lr, gamma)


# ─────────────────────────────────────────────
# 9. STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Epidemic RL", layout="wide")
st.title("🛡️ Epidemic Control — Multi-Algorithm RL Dashboard")

if "trained_agent" not in st.session_state:
    st.session_state["trained_agent"] = None  # (agent, algo_name)
if "trained_agents" not in st.session_state:
    st.session_state["trained_agents"] = {}  # {algo_name: agent}
if "sim_results" not in st.session_state:
    st.session_state["sim_results"] = None

tab1, tab2, tab3 = st.tabs(
    ["📈 Training Dashboard", "🔬 Live Simulation", "📊 Model Evaluation"]
)

# ── TAB 1: TRAINING ──────────────────────────
with tab1:
    st.header("Train a RL Agent")

    train_mode = st.radio(
        "Training Mode", ["Single Model", "All Models"], horizontal=True
    )

    c1, c2 = st.columns(2)
    with c1:
        if train_mode == "Single Model":
            algo = st.selectbox("Algorithm", list(ALGO_INFO.keys()))
            st.markdown(ALGO_INFO[algo])
        else:
            st.info(
                "🤖 All 5 algorithms will be trained sequentially: **D3QN, CPO, Decision Transformer, DreamerV3, MuZero**"
            )
            algo = None
        agent_type = st.selectbox(
            "Agent Preference", ["balanced", "economy-prioritized"]
        )
        episodes = st.slider("Episodes", 1, 200, 100)
    with c2:
        expl = st.selectbox("Exploration (D3QN only)", ["epsilon-greedy", "boltzmann"])
        lr = st.selectbox("Learning Rate", [1e-3, 5e-4])
        gamma = st.selectbox("Discount Factor γ", [0.99, 0.95])
        if train_mode == "Single Model":
            if algo == "Decision Transformer":
                st.info(
                    "ℹ️ DT is offline RL — it collects experience for the first few episodes, "
                    "then trains the Transformer. Expect slow improvement early on."
                )
            if algo == "CPO":
                st.info(
                    "ℹ️ CPO tracks a safety constraint on daily deaths. Watch the λ multiplier "
                    "adapt when deaths rise."
                )

    def _run_training(algo_name, env, agent, episodes, label_prefix=""):
        """Train a single agent and return reward history. Renders progress live."""
        prog = st.progress(0, text=f"Training {algo_name}…")
        status = st.empty()
        chart = st.empty()
        rews = []
        for ep in range(episodes):
            state = env.reset()
            done = False
            ep_rew = 0.0
            while not done:
                act = agent.select_action(state)
                ns, rew, done, nD = env.step(act)
                cost = nD / ENV_CONFIG["N"]
                agent.step_train(state, act, rew, ns, done, cost=cost)
                state = ns
                ep_rew += rew
            agent.end_episode()
            agent.update_exploration()
            rews.append(ep_rew)
            prog.progress(
                (ep + 1) / episodes, text=f"{label_prefix}Episode {ep + 1}/{episodes}"
            )
            eps_str = f"{agent.eps:.3f}" if hasattr(agent, "eps") else "—"
            lam_str = f" | λ={agent.lam.item():.3f}" if hasattr(agent, "lam") else ""
            status.text(
                f"{algo_name} | Ep {ep + 1}/{episodes} | Reward {ep_rew:.2f}"
                f" | ε {eps_str}{lam_str}"
            )
            if (ep + 1) % 2 == 0 or ep == episodes - 1:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(rews, color="purple", alpha=0.6, label="Reward")
                w = min(10, len(rews))
                if len(rews) >= w:
                    ma = np.convolve(rews, np.ones(w) / w, "valid")
                    ax.plot(
                        range(w - 1, len(rews)),
                        ma,
                        color="orange",
                        lw=2,
                        label=f"{w}-ep MA",
                    )
                ax.set_title(f"Training — {algo_name}")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Cumulative Reward")
                ax.legend()
                ax.grid(True)
                chart.pyplot(fig)
                plt.close(fig)
        prog.empty()
        status.empty()
        return rews

    if train_mode == "Single Model":
        if st.button("🚀 Start Training"):
            env = EpidemicEnv(agent_type=agent_type)
            agent = make_agent(
                algo,
                ENV_CONFIG["state_size"],
                ENV_CONFIG["action_size"],
                lr,
                gamma,
                agent_type,
                expl,
            )
            _run_training(algo, env, agent, episodes)
            st.success(f"✅ Training complete! {algo} policy saved.")
            st.session_state["trained_agent"] = (agent, algo)
            st.session_state["trained_agents"][algo] = agent

    else:  # All Models
        if st.button("🚀 Train All Models"):
            algos = list(ALGO_INFO.keys())
            all_rewards = {}
            for i, algo_name in enumerate(algos):
                st.subheader(f"[{i + 1}/{len(algos)}] {algo_name}")
                env = EpidemicEnv(agent_type=agent_type)
                agent = make_agent(
                    algo_name,
                    ENV_CONFIG["state_size"],
                    ENV_CONFIG["action_size"],
                    lr,
                    gamma,
                    agent_type,
                    expl,
                )
                rews = _run_training(
                    algo_name, env, agent, episodes, label_prefix=f"({algo_name}) "
                )
                all_rewards[algo_name] = rews
                st.session_state["trained_agents"][algo_name] = agent
                # Keep last-trained as the default single agent too
                st.session_state["trained_agent"] = (agent, algo_name)
                st.success(f"✅ {algo_name} done — final reward: {rews[-1]:.2f}")

            # Comparison chart
            st.subheader("📊 All-Model Reward Comparison")
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ["purple", "blue", "green", "orange", "red"]
            for (name, rews), col in zip(all_rewards.items(), colors):
                ax.plot(rews, alpha=0.5, color=col, label=name)
                w = min(10, len(rews))
                if len(rews) >= w:
                    ma = np.convolve(rews, np.ones(w) / w, "valid")
                    ax.plot(range(w - 1, len(rews)), ma, color=col, lw=2)
            ax.set_title("Training Rewards — All Models")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Cumulative Reward")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
            st.success(f"🎉 All {len(algos)} models trained and saved!")

# ── TAB 2: SIMULATION ────────────────────────
with tab2:
    st.header("365-Day Epidemic Simulation")

    sim_mode = st.radio(
        "Simulation Mode", ["Single Agent", "All Trained Models"], horizontal=True
    )

    def _run_sim(ag, ag_name, show_live=True):
        lock_seeds(42)
        env = EpidemicEnv("balanced")
        state = env.reset()
        done = False

        old_eps = getattr(ag, "eps", None) if ag else None
        if old_eps is not None:
            ag.eps = 0.0

        if ag:
            ag.reset_for_eval()
        hist = {"S": [], "E": [], "I": [], "R": [], "D": [], "E_econ": [], "Action": []}

        if show_live:
            prog = st.progress(0)
            status = st.empty()
            gph = st.empty()

        while not done:
            if ag:
                act = ag.select_action(state)
            else:
                act = random.randint(0, 3)
            ns, rew, done, nD = env.step(act)
            for k, v in zip(["S", "E", "I", "R", "D", "E_econ"], ns):
                hist[k].append(v)
            hist["Action"].append(act)
            if show_live:
                prog.progress(env.day / ENV_CONFIG["max_days"])
                status.text(
                    f"Day {env.day} | Lockdown level {act} "
                    f"(ζ={ENV_CONFIG['zeta_values'][act]}) | Reward {rew:.2f}"
                )
                if env.day % 10 == 0 or done:
                    fig = plt.figure(figsize=(12, 8))
                    gs_ = fig.add_gridspec(2, 2)
                    ax1 = fig.add_subplot(gs_[0, 0])
                    ax2 = fig.add_subplot(gs_[0, 1])
                    ax3 = fig.add_subplot(gs_[1, :])
                    days = range(1, len(hist["S"]) + 1)
                    for lbl, col in [
                        ("S", "blue"),
                        ("E", "orange"),
                        ("I", "red"),
                        ("R", "green"),
                        ("D", "black"),
                    ]:
                        ax1.plot(days, hist[lbl], label=lbl, color=col)
                    ax1.set_title("SEIRD Dynamics")
                    ax1.set_ylabel("Population")
                    ax1.legend()
                    ax1.grid(True)
                    ax2.plot(days, hist["E_econ"], color="gold", lw=2)
                    ax2.set_title("Cumulative Economic Output")
                    ax2.set_ylabel("Economy Score")
                    ax2.grid(True)
                    ax3.step(days, hist["Action"], color="purple", lw=2, where="post")
                    ax3.set_title(f"AI Policy — {ag_name}")
                    ax3.set_xlabel("Days")
                    ax3.set_ylabel("Lockdown Level")
                    ax3.set_yticks([0, 1, 2, 3])
                    ax3.grid(True)
                    plt.tight_layout()
                    gph.pyplot(fig)
                    plt.close(fig)
            state = ns
        if show_live:
            prog.empty()
            status.empty()
        if old_eps is not None:
            ag.eps = old_eps

        return hist

    if sim_mode == "Single Agent":
        choice = st.radio("Agent:", ["Random Agent", "Trained Agent"])
        if st.button("▶ Run Simulation"):
            if choice == "Trained Agent" and st.session_state["trained_agent"] is None:
                st.error("⚠️ Train an agent first!")
            else:
                ag, ag_name = (
                    st.session_state["trained_agent"]
                    if choice == "Trained Agent"
                    else (None, "Random")
                )
                hist = _run_sim(ag, ag_name, show_live=True)
                st.success("✅ Simulation complete!")

    else:  # All Trained Models
        trained = st.session_state["trained_agents"]
        if not trained:
            st.warning(
                "⚠️ No models found in 'trained_agents'. Train using 'Single Model' mode first, or use 'All Models' training mode."
            )

        include_random = st.checkbox("Also include Random Agent", value=True)
        if st.button("▶ Run All Simulations"):
            if not trained and not include_random:
                st.error("⚠️ No trained agents found. Train at least one model first!")
            else:
                agents_to_run = list(trained.items())
                if include_random:
                    agents_to_run.append(("Random", None))

                if not agents_to_run:
                    st.error("⚠️ No agents to simulate!")
                else:
                    all_hist = {}

                    for ag_name, ag_inst in agents_to_run:
                        with st.spinner(f"Running simulation for {ag_name}..."):
                            all_hist[ag_name] = _run_sim(
                                ag_inst, ag_name, show_live=False
                            )

                    st.subheader("📊 All-Agent Comparison")
                    colors = ["purple", "blue", "green", "orange", "red", "gray"]

                    col_map = {
                        n: colors[i % len(colors)]
                        for i, n in enumerate(all_hist.keys())
                    }

                    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                    for ag_name, hist in all_hist.items():
                        days = range(1, len(hist["S"]) + 1)
                        col = col_map[ag_name]
                        axes[0].plot(days, hist["I"], color=col, label=ag_name)
                        axes[1].plot(days, hist["D"], color=col, label=ag_name)
                        axes[2].plot(days, hist["E_econ"], color=col, label=ag_name)

                    for ax, title, ylabel in zip(
                        axes,
                        ["Infected Over Time", "Cumulative Deaths", "Economic Output"],
                        ["Infected", "Deaths", "Economy Score"],
                    ):
                        ax.set_title(title)
                        ax.set_xlabel("Days")
                        ax.set_ylabel(ylabel)
                        ax.legend(fontsize=7)
                        ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Summary table
                    st.subheader("📋 Summary Table")
                    rows = []
                    for ag_name, hist in all_hist.items():
                        rows.append(
                            {
                                "Agent": ag_name,
                                "Total Deaths": f"{hist['D'][-1]:.1f}",
                                "Peak Infected": f"{max(hist['I']):.1f}",
                                "Economic Output": f"{hist['E_econ'][-1]:.1f}",
                                "Lockdown Days": sum(
                                    1 for a in hist["Action"] if a > 0
                                ),
                            }
                        )
                    st.dataframe(rows, use_container_width=True)

with tab3:
    st.header("📊 Model Evaluation & Metrics")

    eval_mode = st.radio(
        "Evaluation Mode", ["Single Model", "All Trained Models"], horizontal=True
    )

    def _eval_agent(agent, algo_name):
        lock_seeds(42)
        env = EpidemicEnv(agent_type="balanced")
        state = env.reset()
        agent.reset_for_eval()

        old_eps = getattr(agent, "eps", None)
        if old_eps is not None:
            agent.eps = 0.0

        done = False
        total_reward = 0
        max_infected = 0
        total_deaths = 0
        lockdown_days = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, nD = env.step(action)
            total_reward += reward
            max_infected = max(max_infected, next_state[2])
            total_deaths = next_state[4]
            if action > 0:
                lockdown_days += 1
            state = next_state
        if old_eps is not None:
            agent.eps = old_eps
        return {
            "algo": algo_name,
            "total_reward": total_reward,
            "total_economy": state[5],
            "total_deaths": total_deaths,
            "max_infected": max_infected,
            "lockdown_days": lockdown_days,
            "survival_rate": ((1000 - total_deaths) / 1000) * 100,
        }

    if eval_mode == "Single Model":
        st.write(
            "Analyze the final performance metrics of your trained agent over a full 365-day epidemic cycle."
        )
        if st.session_state.get("trained_agent") is None:
            st.warning(
                "⚠️ Please train an agent in the 'Training Dashboard' first to see its evaluation metrics."
            )
        else:
            agent, algo_name = st.session_state["trained_agent"]
            if st.button(f"Evaluate Trained Model ({algo_name})"):
                with st.spinner(
                    f"Running silent evaluation simulation for {algo_name}..."
                ):
                    m = _eval_agent(agent, algo_name)
                st.subheader(f"Performance Summary: {algo_name} (365 Days)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Cumulative Reward", f"{m['total_reward']:.2f}")
                    st.metric(
                        "Total Economic Output",
                        f"{m['total_economy']:.2f}",
                        help="Max possible is 365.0 (No lockdowns).",
                    )
                with col2:
                    st.metric(
                        "Total Deaths",
                        f"{int(m['total_deaths'])}",
                        help="Total population deceased out of 1000.",
                    )
                    st.metric(
                        "Peak Infected",
                        f"{int(m['max_infected'])}",
                        help="Highest number of people infected on a single day. Lower is better.",
                    )
                with col3:
                    st.metric(
                        "Days in Lockdown",
                        f"{m['lockdown_days']}",
                        help="Number of days the agent chose a lockdown level > 0.",
                    )
                    st.metric("Survival Rate", f"{m['survival_rate']:.1f}%")
                st.success(
                    f"Evaluation complete! The metrics above reflect the {algo_name} policy."
                )

    else:  # All Trained Models
        st.write(
            "Evaluate all trained agents side-by-side over a full 365-day epidemic cycle."
        )
        trained = st.session_state["trained_agents"]
        if not trained:
            st.warning(
                "⚠️ No trained agents found. Please train at least one model first."
            )
        else:
            st.info(
                f"Found **{len(trained)}** trained model(s): {', '.join(trained.keys())}"
            )
            if st.button("📊 Evaluate All Trained Models"):
                results = []
                prog = st.progress(0, text="Evaluating…")
                for i, (algo_name, agent) in enumerate(trained.items()):
                    prog.progress(
                        (i + 1) / len(trained), text=f"Evaluating {algo_name}…"
                    )
                    with st.spinner(f"Running evaluation for {algo_name}…"):
                        m = _eval_agent(agent, algo_name)
                    results.append(m)

                prog.empty()
                st.subheader("📋 Side-by-Side Performance Summary")

                # Metric cards per model
                for m in results:
                    with st.expander(f"**{m['algo']}**", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Total Cumulative Reward", f"{m['total_reward']:.2f}"
                            )
                            st.metric(
                                "Total Economic Output", f"{m['total_economy']:.2f}"
                            )
                        with col2:
                            st.metric("Total Deaths", f"{int(m['total_deaths'])}")
                            st.metric("Peak Infected", f"{int(m['max_infected'])}")
                        with col3:
                            st.metric("Days in Lockdown", f"{m['lockdown_days']}")
                            st.metric("Survival Rate", f"{m['survival_rate']:.1f}%")

                # Comparison bar charts
                st.subheader("📊 Comparative Charts")
                names = [m["algo"] for m in results]
                metrics = [
                    ("Total Reward", [m["total_reward"] for m in results], "steelblue"),
                    ("Total Deaths", [m["total_deaths"] for m in results], "tomato"),
                    ("Peak Infected", [m["max_infected"] for m in results], "orange"),
                    ("Economic Output", [m["total_economy"] for m in results], "gold"),
                    (
                        "Survival Rate %",
                        [m["survival_rate"] for m in results],
                        "mediumseagreen",
                    ),
                    (
                        "Lockdown Days",
                        [m["lockdown_days"] for m in results],
                        "mediumpurple",
                    ),
                ]
                fig, axes = plt.subplots(2, 3, figsize=(16, 8))
                for ax, (title, vals, col) in zip(axes.flatten(), metrics):
                    bars = ax.bar(names, vals, color=col, alpha=0.8)
                    ax.set_title(title)
                    ax.set_ylabel(title)
                    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
                    ax.tick_params(axis="x", rotation=20)
                    ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.success(f"✅ Evaluation complete for {len(results)} model(s)!")
