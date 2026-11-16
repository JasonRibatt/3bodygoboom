"""Exact replay of one pilot run.

Runs data_generator.simulate_encounter itself (same integrate() calls, same classification code), with a
REBOUND Simulation subclass that records every IAS15 step through the heartbeat callback. The heartbeat does
not change the step sequence, so the recorded path is the same one that produced the dataset's label.
"""
import os
import sys
import numpy as np
import rebound

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))       # 3bodygoboom/ (holds data_generator.py)
sys.path.insert(0, REPO)
import data_generator as dg                                    # noqa: E402


class _Recorder:
    """Factory used in place of rebound.Simulation: a normal Simulation with a recording heartbeat."""
    live = []                              # (sim, rec_t, rec_xy, hb) for each simulation created

    def __call__(self, *a, **k):
        sim = rebound.Simulation(*a, **k)
        rec_t, rec_xy = [], []
        def hb(_ptr):
            if sim.N == 3:
                p = sim.particles
                rec_t.append(sim.t)
                rec_xy.append([(p[0].x, p[0].y), (p[1].x, p[1].y), (p[2].x, p[2].y)])
        sim.heartbeat = hb
        _Recorder.live.append((sim, rec_t, rec_xy, hb))   # hb kept alive here
        return sim


class _ReboundProxy:                      # what data_generator sees as `rebound`
    Simulation = _Recorder()
    Collision = rebound.Collision


def replay(a_out, e_out, phase_out, tail=60.0, **kw):
    """Returns dict: label, t_event, ejected/colliding info, t[], xy[T,3,2], t_tail_start."""
    _Recorder.live.clear()
    real = dg.rebound
    dg.rebound = _ReboundProxy
    try:
        res = dg.simulate_encounter(a_out, e_out, phase_out, **kw)   # 6 values (pilot) or 9 (run 2)
        label, traj = res[0], res[1]
    finally:
        dg.rebound = real
    sim, rec_t, rec_xy, _ = _Recorder.live[-1]
    t_event = sim.t
    info = {"label": label, "t_event": t_event, "traj": traj, "ejected": None, "h_event": None}

    if label == "Ejection":
        h = dg.find_bound_hierarchy(sim)
        info["ejected"], info["h_event"] = h["k"], h
        sim.integrate(t_event + tail)     # extra time AFTER the classification, for the visual only
    t = np.array(rec_t)
    xy = np.array(rec_xy)
    # the heartbeat can fire twice at one time (end of an integrate call); keep the last
    keep = np.r_[np.diff(t) > 0, True]
    info["t"], info["xy"] = t[keep], xy[keep]
    return info


def check_against_npz(info, traj_saved):
    """Max |difference| between the replay at t=1..20 and the saved trajectory array (x0,x1,x2,y0,y1,y2,...)."""
    diffs = []
    for k in range(20):
        if not np.isfinite(traj_saved[k]).all():
            break
        j = np.argmin(np.abs(info["t"] - (k + 1)))
        x, y = info["xy"][j, :, 0], info["xy"][j, :, 1]
        diffs.append(np.abs(np.r_[x, y] - traj_saved[k, :6]).max())
    return max(diffs) if diffs else np.nan
