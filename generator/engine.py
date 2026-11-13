"""Reduced-order engine model: throttle in, 8 coupled sensor channels out."""

import numpy as np

from contracts import CHANNELS, DT_S, N_STEPS, RUN_S, FaultLabel, Run

# rng stream ids. Start at 1: [seed, 0] is the same stream as a plain seed.
# Never renumber once datasets exist. Add new streams at the end.
THROTTLE_STREAM, ENGINE_STREAM, NOISE_STREAM, FAULT_STREAM = 1, 2, 3, 4

# Upper limit per channel. Set from the physics before looking at any runs.
REDLINES: dict[str, float] = {
    "chamber_pressure": 120.0,      # bar
    "turbopump_speed": 36000.0,     # rpm
    "turbopump_vibration": 2.4,     # g
    "bearing_temp": 420.0,          # K
    "nozzle_temp": 950.0,           # K
    "coolant_flow": 60.0,           # kg/s
    "feed_pressure": 180.0,         # bar
    "mixture_ratio": 7.2,
}

LEVEL_MIN, LEVEL_MAX = 0.6, 1.0
RAMP_UP = (2.5, 3.5)
STEP_WINDOW = (10.0, 50.0)
STEP_GAP = 8.0
STEP_MIN = 0.1
SLEW = 0.2
SHUTDOWN_WINDOW = (54.0, 56.0)
RAMP_DOWN = (2.0, 3.0)

assert STEP_MIN < (LEVEL_MAX - LEVEL_MIN) / 2, "step redraw could loop forever"
assert RAMP_UP[1] < STEP_WINDOW[0], "steps could start during the startup ramp"
assert STEP_WINDOW[1] - STEP_WINDOW[0] >= STEP_GAP, "window can't fit two steps"
assert STEP_WINDOW[1] + SLEW < SHUTDOWN_WINDOW[0], "last step overlaps shutdown"
assert SHUTDOWN_WINDOW[1] + RAMP_DOWN[1] < RUN_S, "shutdown can't finish before the run ends"

ONSET_WINDOW = (10.0, 30.0)  # s, when bearing wear can start
SUB_THRESHOLD_S = 15.0  # s after onset during which redlines must stay quiet
RATE_RANGE = (0.04, 0.06)  # increase in mu per second, relative to healthy
WEAR_CHANNELS = ("bearing_temp", "turbopump_vibration", "turbopump_speed")

assert ONSET_WINDOW[1] + SUB_THRESHOLD_S < SHUTDOWN_WINDOW[0], "fault window runs into shutdown"

NOMINAL = {
    "n_max": 30000.0,       # rpm at full throttle
    "tau_n": 0.5,           # s
    "tau_bearing": 5.0,     # s
    "tau_nozzle": 3.0,      # s at full coolant flow
    "pf_max": 150.0,        # bar at full speed
    "q_max": 50.0,          # kg/s at full feed pressure
    "pc_max": 100.0,        # bar at full flow
    "vib_max": 2.0,         # g at full speed
    "mr_nom": 6.0,
    "t_rest": 290.0,        # K
    "bearing_rise": 60.0,   # K above rest at full speed
    "nozzle_rise": 500.0,   # K above rest at full chamber pressure and flow
    "mu": 1.0,              # bearing friction, relative to a healthy bearing
}
VARIATION = 0.03
K_VIB = 0.2  # vibration +20% when friction doubles
K_LOAD = 0.02  # speed -2% when friction doubles
OFFSET_MAX = 3.0  # calibration offsets up to this many noise sigmas
NOZZLE_TAU_AMB = 30.0  # s, slow loss to ambient when coolant isn't flowing

SENSOR_NOISE = {
    "chamber_pressure": 0.5,        # bar
    "turbopump_speed": 100.0,       # rpm
    "turbopump_vibration": 0.03,    # g
    "bearing_temp": 1.0,            # K
    "nozzle_temp": 3.0,             # K
    "coolant_flow": 0.25,           # kg/s
    "feed_pressure": 0.75,          # bar
    "mixture_ratio": 0.02,
}


def make_throttle(seed: int) -> np.ndarray:
    """Duty cycle: startup ramp, hold, 1-2 mid-run steps, shutdown. Shape (N_STEPS,)."""
    rng = np.random.default_rng([seed, THROTTLE_STREAM])

    ramp_end = rng.uniform(*RAMP_UP)
    level = rng.uniform(LEVEL_MIN, LEVEL_MAX)
    knots = [(0.0, 0.0), (ramp_end, level)]

    n_steps = rng.integers(1, 3)
    lo, hi = STEP_WINDOW
    first = rng.uniform(lo, hi - STEP_GAP * (n_steps - 1))
    step_times = [first] if n_steps == 1 else [first, rng.uniform(first + STEP_GAP, hi)]
    for t_step in step_times:
        new = rng.uniform(LEVEL_MIN, LEVEL_MAX)
        while abs(new - level) < STEP_MIN:
            new = rng.uniform(LEVEL_MIN, LEVEL_MAX)
        knots += [(t_step, level), (t_step + SLEW, new)]
        level = new

    shutdown_start = rng.uniform(*SHUTDOWN_WINDOW)
    shutdown_end = shutdown_start + rng.uniform(*RAMP_DOWN)
    knots += [(shutdown_start, level), (shutdown_end, 0.0), (RUN_S, 0.0)]

    knot_t, knot_v = zip(*knots)
    t = np.arange(N_STEPS) * DT_S
    return np.interp(t, knot_t, knot_v)


def make_fault(seed: int) -> FaultLabel:
    """Bearing wear: friction rises linearly from a random onset at a random rate."""
    rng = np.random.default_rng([seed, FAULT_STREAM])
    onset_s = rng.uniform(*ONSET_WINDOW) # draw order is part of the contract:
    rate = rng.uniform(*RATE_RANGE) # onset first, then rate. Never reorder

    return FaultLabel(
        kind="bearing_wear",
        onset_s=float(onset_s),
        rate=float(rate),
        channels=WEAR_CHANNELS,
    )


def _algebraic(
    n: float | np.ndarray, wear: float | np.ndarray, p: dict[str, float]
) -> tuple[float | np.ndarray, ...]:
    """Channels with no memory, from normalized speed n and bearing wear (scalars or arrays)."""
    feed = p["pf_max"] * n**2
    flow = p["c_flow"] * np.sqrt(feed)
    chamber = p["pc_max"] * flow / p["q_max"]
    vibration = p["vib_max"] * n**2 * (1 + K_VIB * wear)
    return feed, flow, chamber, vibration


def simulate(throttle: np.ndarray, seed: int, noise: bool = True, fault: FaultLabel | None = None) -> Run:
    """Integrate the engine model with explicit Euler. noise=False turns off only sensor noise."""
    engine_rng = np.random.default_rng([seed, ENGINE_STREAM])
    noise_rng = np.random.default_rng([seed, NOISE_STREAM])

    p = {name: value * engine_rng.uniform(1 - VARIATION, 1 + VARIATION) for name, value in NOMINAL.items()}
    p["c_flow"] = p["q_max"] / np.sqrt(p["pf_max"])
    p["h_bearing"] = 1.0
    p["C_bearing"] = p["tau_bearing"] * p["h_bearing"]

    mu = np.full(N_STEPS, p["mu"])
    if fault is not None:
        t = np.arange(N_STEPS) * DT_S
        mu = p["mu"] * (1 + fault.rate * np.maximum(0.0, t - fault.onset_s))
    wear = mu / p["mu"] - 1.0  # 0 when healthy, 1.0 when friction has doubled

    speed, bearing, nozzle = (np.empty(N_STEPS) for _ in range(3))
    feed, flow, chamber, vibration = (np.empty(N_STEPS) for _ in range(4))
    mixture = np.full(N_STEPS, p["mr_nom"])  # one pump, so the flows can't imbalance and nothing drifts
    speed[0] = 0.0
    bearing[0] = nozzle[0] = p["t_rest"]

    for k in range(1, N_STEPS):
        n = speed[k - 1] / p["n_max"]
        feed[k - 1], flow[k - 1], chamber[k - 1], vibration[k - 1] = _algebraic(n, wear[k - 1], p)

        speed_target = p["n_max"] * throttle[k - 1] * (1 - K_LOAD * wear[k - 1])
        speed[k] = speed[k - 1] + DT_S * (speed_target - speed[k - 1]) / p["tau_n"]

        q_friction = p["h_bearing"] * p["bearing_rise"] * mu[k - 1] * n**2
        q_loss = p["h_bearing"] * (bearing[k - 1] - p["t_rest"])
        bearing[k] = bearing[k - 1] + DT_S * (q_friction - q_loss) / p["C_bearing"]

        pc_n = chamber[k - 1] / p["pc_max"]
        q_n = flow[k - 1] / p["q_max"]
        heat_in = p["nozzle_rise"] * pc_n
        cooling = q_n**0.8 * (nozzle[k - 1] - p["t_rest"])
        ambient = (nozzle[k - 1] - p["t_rest"]) * p["tau_nozzle"] / NOZZLE_TAU_AMB
        nozzle[k] = nozzle[k - 1] + DT_S * (heat_in - cooling - ambient) / p["tau_nozzle"]

    n_last = speed[-1] / p["n_max"]
    feed[-1], flow[-1], chamber[-1], vibration[-1] = _algebraic(n_last, wear[-1], p)

    outputs = {
        "chamber_pressure": chamber,
        "turbopump_speed": speed,
        "turbopump_vibration": vibration,
        "bearing_temp": bearing,
        "nozzle_temp": nozzle,
        "coolant_flow": flow,
        "feed_pressure": feed,
        "mixture_ratio": mixture,
    }

    for name in CHANNELS:
        sigma = SENSOR_NOISE[name]
        outputs[name] = outputs[name] + OFFSET_MAX * sigma * engine_rng.uniform(-1, 1)
        if noise:
            outputs[name] = outputs[name] + noise_rng.normal(0, sigma, N_STEPS)

    sensors = np.column_stack([outputs[name] for name in CHANNELS])
    return Run(throttle=throttle, sensors=sensors, seed=seed, fault=fault)
