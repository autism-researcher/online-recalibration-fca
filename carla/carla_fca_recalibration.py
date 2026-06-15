#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Closed-loop CARLA evaluation of online recalibration for an FCA safety supervisor.

Reproduces the paper's EXACT eight-feature composite risk and the recalibration
arms (fixed / batch / online / online+clamp), plus a fixed-TTC AEB baseline and a
no-supervisor baseline, in a CLOSED LOOP: the supervisor actually controls the ego
vehicle through the Traffic Manager, interventions feed back into the trajectory,
and safety is measured from GROUND TRUTH (a CARLA collision sensor) plus physical
near-miss exposure (time-to-collision), not from the composite-risk surrogate.

A distribution shift is induced mid-episode (extra traffic + more aggressive
Traffic Manager) so the calibration-drift problem the paper studies actually
manifests in closed loop.

Tested against the CARLA 0.9.13 Python API.

INTEGRITY NOTE: this script only MEASURES. Every reported number comes from the
running simulator (collision sensor, vehicle states). Nothing is hard-coded or
fabricated. Run it, then analyse the CSV it writes.

Author: M. B. Hossain (harness scaffold). See README.md for setup and usage.
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import deque

import numpy as np

try:
    import carla
except ImportError:
    sys.stderr.write(
        "\n[ERROR] Could not import the 'carla' module.\n"
        "Install it for CARLA 0.9.13, e.g.:  pip install carla==0.9.13\n"
        "or add the packaged .egg to PYTHONPATH (see README.md).\n\n")
    raise

# =====================================================================
#  EXACT PAPER-3 EIGHT-FEATURE RISK MODEL  (do not change without reason)
#  Order: speed, accel, jerk, steer_var, lane_offset, ttc, headway, density
#  Weights sum to 0.94; ttc and headway are danger-on-LOW (inverted).
# =====================================================================
W = dict(speed=0.08, accel=0.10, jerk=0.10, steer=0.08,
         lane=0.12, ttc=0.25, headway=0.15, density=0.06)
B_SPEED, B_ACCEL, B_JERK, B_STEER = 28.0, 7.0, 12.0, 0.35
B_LANE = 1.5
TTC_MIN, TTC_MAX = 0.5, 8.0
HW_MIN, HW_MAX = 2.0, 60.0
B_DENSITY = 25.0


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def composite_risk(speed, accel, jerk, steer_var, lane_off, ttc, headway, density):
    """Return R in [0, 0.94] using the exact normalisation of the paper."""
    xs = _clamp(speed / B_SPEED)
    xa = _clamp(abs(accel) / B_ACCEL)
    xj = _clamp(abs(jerk) / B_JERK)
    xv = _clamp(steer_var / B_STEER)
    xl = _clamp(abs(lane_off) / B_LANE)
    # danger-on-low (inverted clamped affine)
    xt = _clamp((TTC_MAX - ttc) / (TTC_MAX - TTC_MIN))
    xh = _clamp((HW_MAX - headway) / (HW_MAX - HW_MIN))
    xd = _clamp(density / B_DENSITY)
    return (W['speed'] * xs + W['accel'] * xa + W['jerk'] * xj + W['steer'] * xv +
            W['lane'] * xl + W['ttc'] * xt + W['headway'] * xh + W['density'] * xd)


# =====================================================================
#  RECALIBRATION SUPERVISORS  (one boundary-update rule each)
# =====================================================================
class Supervisor(object):
    """Base: decide whether to intervene at this step given the risk R."""
    name = "base"

    def __init__(self, tau, B0):
        self.tau = tau
        self.B = B0

    def step(self, R, ttc):
        """Return (intervene: bool, B_eff: float)."""
        raise NotImplementedError


class NoneSup(Supervisor):
    name = "none"
    def step(self, R, ttc):
        return False, self.B


class FixedSup(Supervisor):
    name = "fixed"
    def step(self, R, ttc):
        return (R > self.B), self.B


class OnlineSup(Supervisor):
    name = "online"
    def __init__(self, tau, B0, gamma=0.05):
        super(OnlineSup, self).__init__(tau, B0)
        self.gamma = gamma
    def step(self, R, ttc):
        intervene = R > self.B
        self.B = _clamp(self.B + self.gamma * ((1.0 if intervene else 0.0) - self.tau))
        return intervene, self.B


class ClampSup(Supervisor):
    """Fast online update + slow finite-sample-banded safety clamp."""
    name = "clamp"
    def __init__(self, tau, B0, gamma=0.05, margin=0.02, slow_window=2500, seed_vals=None):
        super(ClampSup, self).__init__(tau, B0)
        self.gamma = gamma
        self.margin = margin
        self.slow = deque(maxlen=slow_window)
        if seed_vals is not None:
            self.slow.extend(seed_vals[-slow_window:])
    def step(self, R, ttc):
        self.slow.append(R)
        U = float(np.quantile(self.slow, 1.0 - self.tau)) if len(self.slow) > 20 else self.B
        B_eff = min(self.B, U + self.margin)
        intervene = R > B_eff
        # fast update uses the indicator at the EFFECTIVE boundary
        self.B = _clamp(self.B + self.gamma * ((1.0 if intervene else 0.0) - self.tau))
        return intervene, B_eff


class BatchSup(Supervisor):
    """Periodic batch refit of the (1 - tau) quantile on a sliding window."""
    name = "batch"
    def __init__(self, tau, B0, window=600, period=1500):
        super(BatchSup, self).__init__(tau, B0)
        self.buf = deque(maxlen=window)
        self.period = period
        self.t = 0
    def step(self, R, ttc):
        self.buf.append(R)
        self.t += 1
        if self.t % self.period == 0 and len(self.buf) > 20:
            self.B = float(np.quantile(self.buf, 1.0 - self.tau))
        return (R > self.B), self.B


class AEBSup(Supervisor):
    """Fixed time-to-collision automatic-emergency-braking rule (external baseline)."""
    name = "aeb"
    def __init__(self, tau, B0, ttc_thresh=2.0):
        super(AEBSup, self).__init__(tau, B0)
        self.ttc_thresh = ttc_thresh
    def step(self, R, ttc):
        return (ttc < self.ttc_thresh), self.B


def make_supervisor(arm, tau, B0, gamma, margin, slow_window, calib_R):
    if arm == "none":   return NoneSup(tau, B0)
    if arm == "fixed":  return FixedSup(tau, B0)
    if arm == "online": return OnlineSup(tau, B0, gamma)
    if arm == "clamp":  return ClampSup(tau, B0, gamma, margin, slow_window, calib_R)
    if arm == "batch":  return BatchSup(tau, B0)
    if arm == "aeb":    return AEBSup(tau, B0)
    raise ValueError("unknown arm: %s" % arm)


ARMS = ["none", "fixed", "batch", "online", "clamp", "aeb"]


# =====================================================================
#  CARLA HELPERS
# =====================================================================
def vlen(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def dot2(ax, ay, bx, by):
    return ax * bx + ay * by


def setup_world(client, town, dt):
    # Only (re)load the map if we are not already on it. load_world is slow and a
    # fresh load can exceed the client time-out; reuse the current map if it matches.
    try:
        current = client.get_world().get_map().name
    except Exception:
        current = ""
    if town not in current:
        print("Loading map %s (first load can take a while) ..." % town)
        world = client.load_world(town)
    else:
        print("Already on map %s; reusing it." % current)
        world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    settings.no_rendering_mode = False  # set True for headless speed-up
    world.apply_settings(settings)
    return world


def spawn_background(world, tm, blueprints, spawn_points, n, exclude_idx):
    actors = []
    pts = [p for i, p in enumerate(spawn_points) if i != exclude_idx]
    random.shuffle(pts)
    for tf in pts[:n]:
        bp = random.choice(blueprints)
        v = world.try_spawn_actor(bp, tf)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(v, False)
            tm.ignore_lights_percentage(v, 100)
            actors.append(v)
    return actors


def find_lead(ego, vehicles):
    """Nearest vehicle ahead, roughly in the ego lane. Returns (gap_m, lead_speed_along_fwd) or None."""
    etf = ego.get_transform()
    eloc = etf.location
    fwd = etf.get_forward_vector()
    right = etf.get_right_vector()
    evel = ego.get_velocity()
    best = None
    best_d = 1e9
    for v in vehicles:
        if v.id == ego.id:
            continue
        vloc = v.get_transform().location
        rx, ry = vloc.x - eloc.x, vloc.y - eloc.y
        ahead = dot2(rx, ry, fwd.x, fwd.y)
        if ahead <= 0:
            continue
        lateral = abs(dot2(rx, ry, right.x, right.y))
        if lateral > 2.0:
            continue
        d = math.sqrt(rx * rx + ry * ry)
        if d < best_d and d < 60.0:
            best_d = d
            lvel = v.get_velocity()
            lead_fwd_speed = dot2(lvel.x, lvel.y, fwd.x, fwd.y)
            best = (max(0.1, d - 5.0), lead_fwd_speed)  # ~5 m for vehicle lengths
    return best


def density_within(ego, vehicles, radius=35.0):
    eloc = ego.get_transform().location
    c = 0
    for v in vehicles:
        if v.id == ego.id:
            continue
        dl = v.get_transform().location
        if math.hypot(dl.x - eloc.x, dl.y - eloc.y) < radius:
            c += 1
    return c


def lane_offset(world_map, ego):
    etf = ego.get_transform()
    eloc = etf.location
    wp = world_map.get_waypoint(eloc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return 0.0
    c = wp.transform.location
    right = wp.transform.get_right_vector()
    return abs(dot2(eloc.x - c.x, eloc.y - c.y, right.x, right.y))


# =====================================================================
#  ONE EPISODE
# =====================================================================
def run_episode(client, world, tm, arm, cfg, B0, calib_R, seed, calibrate=False):
    """
    Run one closed-loop episode of one arm. If calibrate=True, runs 'none' mode and
    returns the per-tick risk samples for boundary calibration. Otherwise returns a
    metrics dict.
    """
    world_map = world.get_map()
    blib = world.get_blueprint_library()
    veh_bps = [bp for bp in blib.filter('vehicle.*')
               if int(bp.get_attribute('number_of_wheels')) == 4]
    spawn_points = world_map.get_spawn_points()

    random.seed(seed)
    tm.set_random_device_seed(seed)

    ego_idx = seed % len(spawn_points)
    ego_bp = blib.find('vehicle.tesla.model3')
    ego = None
    for tf in [spawn_points[ego_idx]] + spawn_points:
        ego = world.try_spawn_actor(ego_bp, tf)
        if ego is not None:
            ego_idx = spawn_points.index(tf)
            break
    if ego is None:
        raise RuntimeError("could not spawn ego")
    ego.set_autopilot(True, tm.get_port())
    tm.auto_lane_change(ego, False)
    tm.ignore_lights_percentage(ego, 100)

    # ground-truth collision sensor
    col = {"n": 0}
    col_bp = blib.find('sensor.other.collision')
    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
    col_sensor.listen(lambda e: col.__setitem__("n", col["n"] + 1))

    bg = spawn_background(world, tm, veh_bps, spawn_points,
                          cfg['bg_phase1'], ego_idx)

    # optional: inject a slow lead directly ahead of the ego so there is a real
    # car-following conflict (the forward-collision scenario the supervisor handles)
    if cfg.get('inject_lead'):
        try:
            ewp = world_map.get_waypoint(ego.get_transform().location)
            nxt = ewp.next(25.0) if ewp is not None else []
            if nxt:
                ltf = nxt[0].transform
                ltf.location.z += 0.3
                lead = world.try_spawn_actor(random.choice(veh_bps), ltf)
                if lead is not None:
                    lead.set_autopilot(True, tm.get_port())
                    tm.auto_lane_change(lead, False)
                    tm.ignore_lights_percentage(lead, 100)
                    tm.vehicle_percentage_speed_difference(lead, cfg['lead_slow'])
                    bg.append(lead)
        except Exception:
            pass

    sup = make_supervisor(arm, cfg['tau'], B0, cfg['gamma'],
                          cfg['margin'], cfg['slow_window'], calib_R)

    # nominal (aggressive) and intervention Traffic-Manager set-points
    def set_nominal():
        tm.vehicle_percentage_speed_difference(ego, cfg['nominal_speed_diff'])
        tm.distance_to_leading_vehicle(ego, cfg['nominal_gap'])
    def set_intervene():
        tm.vehicle_percentage_speed_difference(ego, cfg['interv_speed_diff'])
        tm.distance_to_leading_vehicle(ego, cfg['interv_gap'])
    set_nominal()

    dt = cfg['dt']
    W_win = cfg['rate_window']
    jerk_ema_tau = 0.40
    alpha = dt / (jerk_ema_tau + dt)

    steer_hist = deque(maxlen=int(round(1.0 / dt)))  # 1 s of steering
    prev_speed = 0.0
    prev_accel = 0.0
    jerk_f = 0.0

    R_samples = []
    interv_hist = deque(maxlen=W_win)
    rate_dev_acc = []
    min_ttc = 99.0
    ttc_viol = 0
    n_record = 0
    interv_total = 0
    shifted = False

    total = cfg['warmup'] + cfg['episode_len']
    for t in range(total):
        world.tick()

        # ----- induce distribution shift at the midpoint -----
        if (not shifted) and t == cfg['warmup'] + cfg['shift_at']:
            shifted = True
            bg += spawn_background(world, tm, veh_bps, spawn_points,
                                   cfg['bg_phase2_extra'], ego_idx)
            tm.global_percentage_speed_difference(cfg['shift_global_speed_diff'])
            for v in bg:
                tm.auto_lane_change(v, True)  # aggressive cut-ins after shift

        # ----- read ego state -----
        vel = ego.get_velocity()
        speed = vlen(vel)
        accel = (speed - prev_speed) / dt
        jerk = (accel - prev_accel) / dt
        jerk_f = (1 - alpha) * jerk_f + alpha * jerk
        prev_speed, prev_accel = speed, accel
        steer_hist.append(ego.get_control().steer)
        steer_var = float(np.var(steer_hist)) if len(steer_hist) > 2 else 0.0
        loff = lane_offset(world_map, ego)

        vehicles = world.get_actors().filter('vehicle.*')
        lead = find_lead(ego, vehicles)
        if lead is None:
            ttc, headway = TTC_MAX, HW_MAX
        else:
            gap, lead_fwd = lead
            fwd = ego.get_transform().get_forward_vector()
            ego_fwd = dot2(vel.x, vel.y, fwd.x, fwd.y)
            closing = ego_fwd - lead_fwd
            ttc = gap / closing if closing > 0.1 else TTC_MAX
            ttc = _clamp(ttc, 0.0, TTC_MAX)
            headway = gap
        dens = density_within(ego, vehicles)

        R = composite_risk(speed, accel, jerk_f, steer_var, loff, ttc, headway, dens)

        # ----- supervisor decision (skip warm-up) -----
        if t < cfg['warmup']:
            continue
        if calibrate:
            R_samples.append(R)
            continue

        intervene, B_eff = sup.step(R, ttc)
        if intervene:
            set_intervene()
        else:
            set_nominal()

        # ----- record ground-truth + calibration metrics -----
        interv_hist.append(1.0 if intervene else 0.0)
        if intervene:
            interv_total += 1
        if len(interv_hist) == W_win:
            rate_dev_acc.append(abs(np.mean(interv_hist) - cfg['tau']))
        if ttc < min_ttc:
            min_ttc = ttc
        if ttc < cfg['ttc_danger']:
            ttc_viol += 1
        R_samples.append(R)
        n_record += 1

    # cleanup
    col_sensor.stop()
    for a in [col_sensor, ego] + bg:
        try:
            a.destroy()
        except Exception:
            pass

    if calibrate:
        return R_samples

    realized_rate = interv_total / max(1, n_record)
    return dict(
        arm=arm, seed=seed,
        collisions=col["n"],
        min_ttc=round(min_ttc, 3),
        ttc_violation_rate=round(ttc_viol / max(1, n_record), 4),
        realized_rate=round(realized_rate, 4),
        rate_deviation=round(float(np.mean(rate_dev_acc)) if rate_dev_acc else 0.0, 4),
        mean_risk=round(float(np.mean(R_samples)) if R_samples else 0.0, 4),
        steps=n_record,
    )


# =====================================================================
#  MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Closed-loop CARLA FCA recalibration study")
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--tm-port', type=int, default=8000)
    ap.add_argument('--town', default='Town03')
    ap.add_argument('--arms', default='all',
                    help="comma list from %s, or 'all'" % ARMS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--tau', type=float, default=0.10)
    ap.add_argument('--gamma', type=float, default=0.05)
    ap.add_argument('--margin', type=float, default=0.02)
    ap.add_argument('--dt', type=float, default=0.05, help="fixed timestep (0.05=20Hz, 0.04=25Hz)")
    ap.add_argument('--episode-len', type=int, default=2400, help="recorded ticks (~120 s at 20 Hz)")
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--bg', type=int, default=50, help="background vehicles, phase 1")
    ap.add_argument('--no-inject-lead', action='store_true',
                    help="do NOT spawn a slow lead ahead of the ego")
    ap.add_argument('--out', default='carla_results.csv')
    args = ap.parse_args()

    arms = ARMS if args.arms == 'all' else args.arms.split(',')

    cfg = dict(
        tau=args.tau, gamma=args.gamma, margin=args.margin, dt=args.dt,
        episode_len=args.episode_len, warmup=args.warmup,
        shift_at=args.episode_len // 2,            # shift at the midpoint
        rate_window=max(60, int(6.0 / args.dt)),   # ~6 s window
        slow_window=2500,
        ttc_danger=1.5,                            # physical near-miss threshold
        bg_phase1=args.bg,
        bg_phase2_extra=max(10, args.bg // 2),     # extra traffic after shift
        shift_global_speed_diff=-30.0,             # background drives ~30% faster after shift
        nominal_speed_diff=-20.0,                  # moderately aggressive nominal driver (~20% over limit)
        nominal_gap=1.2,                           # follows close but not permanently crashing
        interv_speed_diff=50.0,                    # strong 50% slowdown when supervisor engages
        interv_gap=12.0,
        inject_lead=(not args.no_inject_lead),     # spawn a slow lead ahead of the ego
        lead_slow=40.0,                            # injected lead drives 40% slower (ego approaches, not pinned)
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    try:
        sv = client.get_server_version()
        print("Connected to CARLA server %s at %s:%d" % (sv, args.host, args.port))
    except RuntimeError:
        sys.stderr.write(
            "\n[ERROR] Could not reach the CARLA server at %s:%d.\n"
            "  1. Start it first:  CarlaUE4.exe -quality-level=Epic -carla-rpc-port=%d\n"
            "  2. Wait until the 3D window shows a city scene (first boot compiles\n"
            "     shaders and can take several minutes).\n"
            "  3. Check the port matches (--port) and no firewall blocks it.\n\n"
            % (args.host, args.port, args.port))
        raise
    world = setup_world(client, args.town, args.dt)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)

    rows = []
    try:
        for seed in range(1000, 1000 + args.seeds):
            # ---- Stage A: calibrate the boundary in 'none' mode on this seed ----
            print("[seed %d] calibrating boundary ..." % seed)
            calib_R = run_episode(client, world, tm, "none", cfg, 0.5, None, seed, calibrate=True)
            B0 = float(np.quantile(calib_R, 1.0 - cfg['tau']))
            print("[seed %d] B0 = %.4f  (%d calibration ticks)" % (seed, B0, len(calib_R)))

            # ---- Stage B: each arm under the SAME seed (paired) ----
            for arm in arms:
                m = run_episode(client, world, tm, arm, cfg, B0, calib_R, seed)
                m['B0'] = round(B0, 4)
                rows.append(m)
                print("  %-7s coll=%d  minTTC=%.2f  ttc<1.5=%.3f  rate=%.3f  rate_dev=%.4f"
                      % (arm, m['collisions'], m['min_ttc'], m['ttc_violation_rate'],
                         m['realized_rate'], m['rate_deviation']))
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)
        tm.set_synchronous_mode(False)

    # ---- write CSV ----
    keys = ['arm', 'seed', 'B0', 'collisions', 'min_ttc', 'ttc_violation_rate',
            'realized_rate', 'rate_deviation', 'mean_risk', 'steps']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print("\nWrote %s (%d rows)." % (args.out, len(rows)))

    # ---- per-arm summary across seeds ----
    print("\n=== per-arm means across seeds ===")
    print("%-7s %8s %8s %10s %8s %9s" %
          ("arm", "coll", "minTTC", "ttc<1.5", "rate", "rate_dev"))
    for arm in arms:
        a = [r for r in rows if r['arm'] == arm]
        if not a:
            continue
        f = lambda k: np.mean([r[k] for r in a])
        print("%-7s %8.2f %8.2f %10.3f %8.3f %9.4f" %
              (arm, f('collisions'), f('min_ttc'), f('ttc_violation_rate'),
               f('realized_rate'), f('rate_deviation')))


if __name__ == '__main__':
    main()
