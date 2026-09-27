#!/usr/bin/env python3
"""
Controlled CARLA conflict scenarios in which the supervisor is the operative safety
layer (the ego is not on autopilot; the only emergency braking comes from the
supervisor). These are the scripted conflicts reported in the paper (cs_*_b.csv).

Scenario handling:
  * vehicles follow the lane (pure-pursuit steering);
  * hazards spawn on the lane ahead, with a world-coordinate fallback;
  * the cut-in is a brief, bounded lateral maneuver;
  * the collision sensor counts only collisions with the hazard (a vehicle or walker),
    not walls or curbs;
  * the spectator camera follows the ego.

USAGE (short pilot first):
  python carla_conflict_scenarios.py --scenario lead_brake --seeds 0-1 --episode 12 --out pilot.csv
"""
import argparse, csv, math, sys
from collections import deque
import numpy as np

try:
    import carla
except Exception as e:                       # pragma: no cover
    carla = None
    _IMPORT_ERR = e

from carla_fca_recalibration import (
    composite_risk, make_supervisor, setup_world, density_within, lane_offset,
    vlen, TTC_MAX, HW_MAX,
)

DT          = 0.05
TARGET_KMH  = 50.0
WARMUP_S    = 4.0
ARMS        = ["fixed", "online", "clamp", "aeb"]
TAU, GAMMA, MARGIN, SLOW = 0.10, 0.05, 0.02, 2500
HARD_BRAKE  = 1.0
JERK_TAU    = 0.40


# ----------------------------------------------------------------------
# Kinematics to the scripted hazard (vehicle OR walker)
# ----------------------------------------------------------------------
def hazard_kinematics(ego, hazard):
    et = ego.get_transform(); el = et.location
    hl = hazard.get_transform().location
    fwd = et.get_forward_vector()
    dx, dy = hl.x - el.x, hl.y - el.y
    gap = dx * fwd.x + dy * fwd.y - 4.5
    ev, hv = ego.get_velocity(), hazard.get_velocity()
    closing = (ev.x - hv.x) * fwd.x + (ev.y - hv.y) * fwd.y
    if gap <= 0.0:
        return max(gap, -1.0), closing, 0.0
    ttc = gap / closing if closing > 0.1 else TTC_MAX
    return gap, closing, max(0.0, min(ttc, TTC_MAX))


def drac_of(gap, closing):
    return (closing ** 2) / (2.0 * gap) if (gap > 0.1 and closing > 0.1) else 0.0


def lane_steer(world_map, vehicle, lookahead=6.0):
    """Pure-pursuit steering that keeps a vehicle following its driving lane.
    Crash-proof: returns 0.0 on any failure. If vehicles steer the WRONG way,
    flip the sign of the return value (one line)."""
    try:
        tf = vehicle.get_transform()
        wp = world_map.get_waypoint(tf.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0
        nxt = wp.next(lookahead)
        if not nxt:
            return 0.0
        tgt = nxt[0].transform.location
        veh_yaw = math.radians(tf.rotation.yaw)
        des = math.atan2(tgt.y - tf.location.y, tgt.x - tf.location.x)
        err = (des - veh_yaw + math.pi) % (2.0 * math.pi) - math.pi
        return max(-1.0, min(1.0, err / (math.pi / 4.0)))     # <-- flip sign here if needed
    except Exception:
        return 0.0


def _ahead_wp(world_map, loc, dist):
    try:
        wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return None
        nxt = wp.next(dist)
        return nxt[0] if nxt else None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Per-tick composite risk (mirrors run_episode in the main harness)
# ----------------------------------------------------------------------
class FeatureState:
    def __init__(self):
        self.prev_speed = 0.0; self.prev_accel = 0.0; self.jerk_f = 0.0
        self.steer_hist = deque(maxlen=int(round(1.0 / DT)))

    def risk(self, ego, world_map, vehicles, ttc, headway):
        v = ego.get_velocity(); speed = vlen(v)
        accel = (speed - self.prev_speed) / DT
        jerk = (accel - self.prev_accel) / DT
        a = DT / (JERK_TAU + DT)
        self.jerk_f = (1 - a) * self.jerk_f + a * jerk
        self.prev_speed, self.prev_accel = speed, accel
        self.steer_hist.append(ego.get_control().steer)
        steer_var = float(np.var(self.steer_hist)) if len(self.steer_hist) > 2 else 0.0
        loff = lane_offset(world_map, ego)
        dens = density_within(ego, vehicles)
        return composite_risk(speed, accel, self.jerk_f, steer_var, loff, ttc, headway, dens), speed


# ----------------------------------------------------------------------
# Spawning
# ----------------------------------------------------------------------
def spawn_ego(world, bp, spawn, seed):
    ego_bp = bp.find("vehicle.tesla.model3")
    pts = [spawn[seed % len(spawn)]] + spawn
    for tf in pts:
        ego = world.try_spawn_actor(ego_bp, tf)
        if ego is not None:
            ego.set_autopilot(False)
            return ego, tf
    raise RuntimeError("could not spawn ego")


def _spawn_vehicle_ahead(world, vbp, world_map, sp, lateral=0.0):
    """Spawn a vehicle ~25 m ahead on the lane, with a world-coordinate fallback so a
    car always appears. `lateral` shifts it sideways (for the cut-in)."""
    awp = _ahead_wp(world_map, sp.location, 25.0)
    candidates = []
    if awp is not None:
        if lateral != 0.0:
            lane = awp.get_left_lane()
            if (lane is not None and lane.lane_type == carla.LaneType.Driving
                    and lane.lane_id * awp.lane_id > 0):
                awp = lane
        tf = awp.transform; tf.location.z += 0.3
        candidates.append(tf)
    # fallback: straight world coordinates ahead of the spawn point
    fwd = sp.get_forward_vector(); rgt = sp.get_right_vector()
    for d in (25.0, 18.0, 12.0):
        loc = carla.Location(sp.location.x + d * fwd.x + lateral * rgt.x,
                             sp.location.y + d * fwd.y + lateral * rgt.y,
                             sp.location.z + 0.3)
        candidates.append(carla.Transform(loc, sp.rotation))
    for tf in candidates:
        h = world.try_spawn_actor(vbp, tf)
        if h is not None:
            return h
    return None


def build_hazard(world, bp, sp, name, trigger_s, world_map):
    """Spawn the scripted hazard and return (actor, drive_fn, is_walker); (None,..) on failure."""
    vbp = [b for b in bp.filter("vehicle.*")
           if int(b.get_attribute("number_of_wheels")) == 4][0]

    if name == "lead_brake":
        h = _spawn_vehicle_ahead(world, vbp, world_map, sp, lateral=0.0)
        if h is None:
            return None, None, False
        h.set_autopilot(False)
        def drive(t, h=h):
            st = lane_steer(world_map, h)
            if t >= trigger_s:
                h.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=st))
            else:
                h.apply_control(carla.VehicleControl(throttle=0.45, steer=st))
        return h, drive, False

    if name == "cut_in":
        h = _spawn_vehicle_ahead(world, vbp, world_map, sp, lateral=3.5)
        if h is None:
            return None, None, False
        h.set_autopilot(False)
        def drive(t, h=h):
            st = lane_steer(world_map, h)
            if trigger_s <= t < trigger_s + 1.4:      # brief, bounded cut toward the ego lane
                st = -0.35
            h.apply_control(carla.VehicleControl(throttle=0.5, steer=float(max(-1.0, min(1.0, st)))))
        return h, drive, False

    if name == "crossing":
        wbp = bp.filter("walker.pedestrian.*")[0]
        awp = _ahead_wp(world_map, sp.location, 25.0)
        if awp is not None:
            rt = awp.transform; right = rt.get_right_vector(); base = rt.location
        else:
            right = sp.get_right_vector(); fwd = sp.get_forward_vector()
            base = carla.Location(sp.location.x + 25 * fwd.x, sp.location.y + 25 * fwd.y, sp.location.z)
        h = None
        for d in (4.0, 6.0, 2.0, 8.0, -4.0):
            wl = carla.Location(base.x + d * right.x, base.y + d * right.y, base.z + 1.0)
            h = world.try_spawn_actor(wbp, carla.Transform(wl))
            if h is not None:
                break
        if h is None:
            return None, None, True
        dirx, diry = -right.x, -right.y               # walk across the ego path
        def drive(t, h=h, dirx=dirx, diry=diry):
            if t >= trigger_s:
                c = carla.WalkerControl(); c.speed = 1.6
                c.direction = carla.Vector3D(dirx, diry, 0.0)
                h.apply_control(c)
        return h, drive, True
    raise ValueError(name)


def follow_with_camera(world, ego):
    try:
        sp = world.get_spectator()
        etf = ego.get_transform(); f = etf.get_forward_vector()
        loc = carla.Location(etf.location.x - 8 * f.x, etf.location.y - 8 * f.y, etf.location.z + 6)
        sp.set_transform(carla.Transform(loc, carla.Rotation(pitch=-20, yaw=etf.rotation.yaw)))
    except Exception:
        pass


# ----------------------------------------------------------------------
# One episode (calibrate=True returns R samples for B0; else metrics)
# ----------------------------------------------------------------------
def run_episode(world, bp, spawn, scenario, arm, seed, B0, calib_R,
                episode_s, trigger_s, calibrate=False):
    ego, sp = spawn_ego(world, bp, spawn, seed)
    world_map = world.get_map()
    hazard = drive = None; is_walker = False
    if not calibrate:
        hazard, drive, is_walker = build_hazard(world, bp, sp, scenario, trigger_s, world_map)
        if hazard is None:
            try: ego.destroy()
            except Exception: pass
            return {"scenario": scenario, "arm": arm, "seed": seed,
                    "collision": "", "min_ttc": "", "max_drac": "", "pet": ""}

    blib = world.get_blueprint_library()
    col = {"n": 0}
    col_sensor = world.spawn_actor(blib.find("sensor.other.collision"),
                                   carla.Transform(), attach_to=ego)
    def _on_collision(e):                          # count ONLY hits with the hazard
        oa = getattr(e, "other_actor", None)       # (a vehicle or walker), not walls/curbs
        tid = oa.type_id if oa is not None else ""
        if tid.startswith("vehicle.") or tid.startswith("walker."):
            col["n"] += 1
    col_sensor.listen(_on_collision)

    sup = None if calibrate else make_supervisor(arm, TAU, B0, GAMMA, MARGIN, SLOW, calib_R)
    fs = FeatureState()
    target = TARGET_KMH / 3.6
    R_samples = []
    min_ttc, max_drac, pet = TTC_MAX, 0.0, None
    cross_in = cross_out = None
    n_steps = int((WARMUP_S + episode_s) / DT)

    try:
        for k in range(n_steps):
            t = k * DT - WARMUP_S
            if drive is not None and t >= 0:
                drive(t)
            follow_with_camera(world, ego)
            vehicles = world.get_actors().filter("vehicle.*")
            if hazard is not None and not is_walker:
                gap, closing, ttc = hazard_kinematics(ego, hazard)
                headway = max(0.0, gap)
            elif hazard is not None and is_walker:
                gap, closing, ttc = hazard_kinematics(ego, hazard)
                headway = max(0.0, gap)
            else:
                ttc, headway, gap, closing = TTC_MAX, HW_MAX, HW_MAX, 0.0
            R, speed = fs.risk(ego, world_map, vehicles, ttc, headway)

            est = lane_steer(world_map, ego)             # keep the ego in its lane
            if calibrate:
                if t >= 0:
                    R_samples.append(R)
                ego.apply_control(carla.VehicleControl(throttle=0.45, steer=est))
                world.tick(); continue

            fire, _ = sup.step(R, ttc)
            if fire:
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=HARD_BRAKE, steer=est))
            else:
                ego.apply_control(carla.VehicleControl(
                    throttle=float(max(0.0, min(0.7, 0.3 + 0.1 * (target - speed)))), brake=0.0, steer=est))
            world.tick()

            if t >= 0:
                min_ttc = min(min_ttc, ttc)
                max_drac = max(max_drac, drac_of(gap, closing))
                if is_walker:
                    wl = hazard.get_transform().location; el = ego.get_transform().location
                    in_zone = abs(wl.x - el.x) < 3.0
                    if in_zone and cross_in is None: cross_in = t
                    if cross_in is not None and not in_zone and cross_out is None: cross_out = t
        if cross_in is not None and cross_out is not None:
            pet = cross_out - cross_in
    finally:
        try: col_sensor.stop()
        except Exception: pass
        for a in (col_sensor, hazard, ego):
            try:
                if a is not None: a.destroy()
            except Exception: pass

    if calibrate:
        return R_samples
    return {"scenario": scenario, "arm": arm, "seed": seed,
            "collision": int(col["n"] > 0), "min_ttc": round(min_ttc, 3),
            "max_drac": round(max_drac, 3), "pet": ("" if pet is None else round(pet, 3))}


def parse_seeds(spec):
    if "-" in spec:
        a, b = spec.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    if carla is None:
        sys.exit("CARLA not importable: %s\nRun on a machine with CARLA 0.9.13." % _IMPORT_ERR)
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["lead_brake", "cut_in", "crossing"])
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--episode", type=float, default=20.0)
    ap.add_argument("--trigger", type=float, default=8.0)
    ap.add_argument("--town", default="Town03")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client = carla.Client("localhost", 2000); client.set_timeout(60.0)
    world = setup_world(client, args.town, DT)
    bp = world.get_blueprint_library()
    spawn = world.get_map().get_spawn_points()

    print("calibrating B0 (free drive)...")
    calib_R = run_episode(world, bp, spawn, args.scenario, "none", 0, None, None,
                          args.episode, args.trigger, calibrate=True)
    B0 = float(np.quantile(calib_R, 1.0 - TAU)) if calib_R else 0.5
    print("B0 = %.4f from %d samples" % (B0, len(calib_R)))

    rows = []
    for seed in parse_seeds(args.seeds):
        for arm in ARMS:
            m = run_episode(world, bp, spawn, args.scenario, arm, seed, B0, calib_R,
                            args.episode, args.trigger)
            rows.append(m)
            if m["collision"] == "":
                print("seed %2d %-7s SKIPPED (hazard spawn failed)" % (seed, arm))
            else:
                print("seed %2d %-7s coll=%d minTTC=%.2f maxDRAC=%.2f pet=%s"
                      % (seed, arm, m["collision"], m["min_ttc"], m["max_drac"], m["pet"]))

    out = args.out or ("cs_%s.csv" % args.scenario)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "arm", "seed", "collision",
                                          "min_ttc", "max_drac", "pet"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print("\nwrote", out)
    for arm in ARMS:
        a = [r for r in rows if r["arm"] == arm and r["collision"] != ""]
        if a:
            print("  %-7s coll_rate=%.3f mean_min_ttc=%.2f (n=%d valid)"
                  % (arm, sum(int(r["collision"]) for r in a) / len(a),
                     sum(float(r["min_ttc"]) for r in a) / len(a), len(a)))


if __name__ == "__main__":
    main()
