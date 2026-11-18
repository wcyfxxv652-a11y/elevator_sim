
"""
Elevator group simulation
=========================

A discrete-time (1 second per tick) simulation of a bank of elevators
serving an office building over a working day:

* time-varying passenger traffic: morning up-peak, lunch, evening
  down-peak, plus background inter-floor trips (Poisson arrivals);
* "collective control" cars: keep travelling while there are requests
  ahead, then reverse; capacity limit; door and boarding times;
* a cost-based group dispatcher that hands every hall call to the car
  expected to reach it soonest.

Every passenger and every car position is logged, collected into pandas
DataFrames, summarised, exported to CSV and plotted with matplotlib.

Examples
--------
    python elevator_sim.py                      # 12 floors, 4 cars, 07:00-19:00
    python elevator_sim.py --elevators 3 --floors 20
    python elevator_sim.py --compare            # sweep the number of cars
    python elevator_sim.py --animate            # watch the busiest 15 minutes
    python elevator_sim.py --no-show            # only write files
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

UP, DOWN, NONE = 1, -1, 0
IDLE, MOVING, DOORS = "idle", "moving", "doors"
BASE_DATE = pd.Timestamp("2026-01-05")  # an arbitrary Monday, for clock times


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class Config:
    floors: int = 12          # floor 0 is the lobby
    elevators: int = 4
    capacity: int = 10        # passengers per car
    travel_time: int = 2      # seconds per floor
    door_time: int = 4        # seconds to open + close doors
    board_time: int = 1       # seconds per passenger boarding / alighting
    start_hour: float = 7.0
    hours: float = 12.0
    traffic_scale: float = 1.0
    seed: int = 42


@dataclass
class Passenger:
    pid: int
    arrival: int
    origin: int
    dest: int
    board: int = -1
    alight: int = -1
    car: int = -1

    @property
    def direction(self) -> int:
        return UP if self.dest > self.origin else DOWN


@dataclass
class Car:
    cid: int
    floor: int = 0
    direction: int = NONE
    state: str = IDLE
    timer: int = 0
    riders: list = field(default_factory=list)
    car_stops: set = field(default_factory=set)    # floors riders asked for
    hall_stops: set = field(default_factory=set)   # (floor, dir) calls assigned to us

    def targets(self) -> set:
        return self.car_stops | {f for f, _ in self.hall_stops}

    def has_targets_ahead(self, d: int) -> bool:
        return d != NONE and any((f - self.floor) * d > 0 for f in self.targets())

    def position(self, travel_time: int) -> float:
        if self.state == MOVING:
            return self.floor + self.direction * (1 - self.timer / travel_time)
        return float(self.floor)


def traffic_profile(hour: float) -> tuple[float, np.ndarray]:
    """Arrival rate (passengers/min, 12-floor building) and the mix of
    trip types [incoming from lobby, outgoing to lobby, inter-floor]."""
    def bump(mu, sigma):
        return np.exp(-((hour - mu) / sigma) ** 2)

    morning, lunch, evening = bump(8.75, 0.6), bump(12.5, 0.7), bump(17.25, 0.6)
    rate = 1.5 + 14 * morning + 7 * lunch + 12 * evening
    w_in = 0.3 + 9 * morning + (3 * lunch if hour > 12.5 else 0)
    w_out = 0.3 + 9 * evening + (3 * lunch if hour <= 12.5 else 0)
    mix = np.array([w_in, w_out, 1.0])
    return rate, mix / mix.sum()


class Simulation:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.cars = [Car(i) for i in range(cfg.elevators)]
        self.queues = {(f, d): [] for f in range(cfg.floors) for d in (UP, DOWN)}
        self.assigned: dict[tuple[int, int], int] = {}
        self.passengers: list[Passenger] = []
        self.trace: list[tuple] = []
        self.queue_trace: list[tuple] = []
        self.delivered = 0
        self.t = 0

    # ---- traffic ---------------------------------------------------------
    def spawn(self) -> None:
        cfg = self.cfg
        rate, mix = traffic_profile(cfg.start_hour + self.t / 3600)
        lam = rate * cfg.traffic_scale * (cfg.floors - 1) / 11 / 60
        top = cfg.floors - 1
        for _ in range(self.rng.poisson(lam)):
            kind = self.rng.choice(3, p=mix)
            if kind == 0:
                o, d = 0, int(self.rng.integers(1, top + 1))
            elif kind == 1:
                o, d = int(self.rng.integers(1, top + 1)), 0
            else:
                o, d = (int(x) for x in self.rng.choice(cfg.floors, 2, replace=False))
            p = Passenger(len(self.passengers), self.t, o, d)
            self.passengers.append(p)
            self.queues[(o, p.direction)].append(p)

    # ---- group dispatcher ------------------------------------------------
    def cost(self, car: Car, f: int, d: int) -> float:
        """Rough estimate of seconds until `car` could pick up call (f, d)."""
        cfg = self.cfg
        pos = car.position(cfg.travel_time)
        targets = car.targets()
        stop_penalty = cfg.door_time + 2 * cfg.board_time
        if not targets and car.state != MOVING:
            c = abs(pos - f) * cfg.travel_time
        else:
            cd = car.direction or (UP if f > pos else DOWN)
            ahead = (f - pos) * cd
            on_the_way = cd == d and (ahead > 0 or (ahead == 0 and car.state != MOVING))
            if on_the_way:
                travel = ahead
                stops = sum(0 < (x - pos) * cd < ahead for x in targets)
            else:  # finish the current run, then come back
                reach = max([(x - pos) * cd for x in targets] + [0.0])
                travel = reach + abs(pos + reach * cd - f)
                stops = len(targets)
            c = travel * cfg.travel_time + stops * stop_penalty
        if len(car.riders) >= cfg.capacity:
            c += 600
        return c

    def dispatch(self) -> None:
        for key, q in self.queues.items():
            if q and key not in self.assigned:
                best = min(self.cars, key=lambda c: self.cost(c, *key))
                self.assigned[key] = best.cid
                best.hall_stops.add(key)

    def release(self, key: tuple[int, int]) -> None:
        cid = self.assigned.pop(key, None)
        if cid is not None:
            self.cars[cid].hall_stops.discard(key)

    def drop_stale(self, car: Car) -> None:
        for key in [k for k in car.hall_stops if not self.queues[k]]:
            self.release(key)

    # ---- car controller --------------------------------------------------
    def choose_direction(self, car: Car) -> int:
        d = car.direction
        if car.has_targets_ahead(d):
            return d
        here = [hd for f, hd in car.hall_stops if f == car.floor]
        if here:
            return d if d in here else here[0]
        if car.has_targets_ahead(-d):
            return -d
        targets = car.targets()
        if not targets:
            return NONE
        nearest = min(targets, key=lambda f: (abs(f - car.floor), f))
        return UP if nearest > car.floor else DOWN if nearest < car.floor else NONE

    def serviceable_here(self, car: Car) -> bool:
        if car.floor in car.car_stops:
            return True
        key = (car.floor, car.direction)
        return (car.direction != NONE and key in car.hall_stops
                and bool(self.queues[key]) and len(car.riders) < self.cfg.capacity)

    def board(self, car: Car) -> int:
        if car.direction == NONE:
            return 0
        key = (car.floor, car.direction)
        q = self.queues[key]
        space = self.cfg.capacity - len(car.riders)
        if not q or space <= 0:
            return 0
        boarding = q[:space]
        del q[:space]
        for p in boarding:
            p.board, p.car = self.t, car.cid
            car.riders.append(p)
            car.car_stops.add(p.dest)
        # Served, or the car is full and the rest need another car.
        self.release(key)
        return len(boarding)

    def open_doors(self, car: Car) -> None:
        leaving = [p for p in car.riders if p.dest == car.floor]
        for p in leaving:
            p.alight = self.t
            car.riders.remove(p)
        self.delivered += len(leaving)
        car.car_stops.discard(car.floor)
        self.drop_stale(car)
        car.direction = self.choose_direction(car)
        boarded = self.board(car)
        car.state = DOORS
        car.timer = self.cfg.door_time + (len(leaving) + boarded) * self.cfg.board_time

    def decide(self, car: Car) -> None:
        """Called whenever a car is at a floor and free to act."""
        self.drop_stale(car)
        car.direction = self.choose_direction(car)
        if self.serviceable_here(car):
            self.open_doors(car)
            return
        for key in [k for k in car.hall_stops if k[0] == car.floor]:
            self.release(key)  # can't serve it from here (e.g. full); re-dispatch
        if not car.has_targets_ahead(car.direction):
            targets = car.targets()
            if not targets:
                car.direction, car.state = NONE, IDLE
                return
            nearest = min(targets, key=lambda f: (abs(f - car.floor), f))
            car.direction = UP if nearest > car.floor else DOWN
        car.state, car.timer = MOVING, self.cfg.travel_time

    def step_car(self, car: Car) -> None:
        if car.state == MOVING:
            car.timer -= 1
            if car.timer > 0:
                return
            car.floor += car.direction
            car.state = IDLE
            self.decide(car)
        elif car.state == DOORS:
            car.timer -= 1
            if car.timer > 0:
                return
            late = self.board(car)  # people who arrived while the doors were open
            if late:
                car.timer = late * self.cfg.board_time
                return
            car.state = IDLE
            self.decide(car)
        else:
            self.decide(car)

    # ---- main loop -------------------------------------------------------
    def run(self, max_drain: int = 3600) -> "Results":
        horizon = int(self.cfg.hours * 3600)
        tt = self.cfg.travel_time
        while True:
            if self.t < horizon:
                self.spawn()
            elif self.delivered == len(self.passengers) or self.t >= horizon + max_drain:
                break
            self.dispatch()
            for car in self.cars:
                self.step_car(car)
            for car in self.cars:
                self.trace.append((self.t, car.cid, car.position(tt), len(car.riders), car.state))
            self.queue_trace.append((self.t, sum(len(q) for q in self.queues.values())))
            self.t += 1
        return Results.from_sim(self)


# --------------------------------------------------------------------------
# Analysis (pandas)
# --------------------------------------------------------------------------
@dataclass
class Results:
    cfg: Config
    passengers: pd.DataFrame
    trace: pd.DataFrame
    queue: pd.DataFrame

    @classmethod
    def from_sim(cls, sim: Simulation) -> "Results":
        cfg = sim.cfg
        t0 = BASE_DATE + pd.to_timedelta(cfg.start_hour, unit="h")

        pax = pd.DataFrame([vars(p) for p in sim.passengers])
        for col in ("board", "alight", "car"):
            pax[col] = pax[col].where(pax[col] >= 0)
        pax["direction"] = np.where(pax["dest"] > pax["origin"], "up", "down")
        pax["trip"] = np.select([pax["origin"] == 0, pax["dest"] == 0],
                                ["incoming", "outgoing"], "interfloor")
        pax["wait"] = pax["board"] - pax["arrival"]
        pax["ride"] = pax["alight"] - pax["board"]
        pax["journey"] = pax["alight"] - pax["arrival"]
        pax["time"] = t0 + pd.to_timedelta(pax["arrival"], unit="s")

        trace = pd.DataFrame(sim.trace, columns=["t", "car", "position", "load", "state"])
        trace["time"] = t0 + pd.to_timedelta(trace["t"], unit="s")
        queue = pd.DataFrame(sim.queue_trace, columns=["t", "waiting"])
        queue["time"] = t0 + pd.to_timedelta(queue["t"], unit="s")
        return cls(cfg, pax, trace, queue)

    def summary(self) -> pd.Series:
        p = self.passengers
        w = p["wait"].dropna()
        return pd.Series({
            "passengers": len(p),
            "delivered %": 100 * p["alight"].notna().mean(),
            "mean wait (s)": w.mean(),
            "median wait (s)": w.median(),
            "95th pct wait (s)": w.quantile(0.95),
            "max wait (s)": w.max(),
            "waits > 60 s (%)": 100 * (w > 60).mean(),
            "mean ride (s)": p["ride"].mean(),
            "mean journey (s)": p["journey"].mean(),
            "car busy (%)": 100 * (self.trace["state"] != IDLE).mean(),
            "mean car load": self.trace["load"].mean(),
        })

    def by_period(self, freq: str = "15min") -> pd.DataFrame:
        return (self.passengers.set_index("time")
                .resample(freq)
                .agg(arrivals=("pid", "size"),
                     mean_wait=("wait", "mean"),
                     p95_wait=("wait", lambda s: s.quantile(0.95)),
                     mean_journey=("journey", "mean")))

    def utilization(self) -> pd.DataFrame:
        u = (self.trace.groupby(["car", "state"]).size()
             .unstack(fill_value=0)
             .reindex(columns=[MOVING, DOORS, IDLE], fill_value=0))
        u = 100 * u.div(u.sum(axis=1), axis=0)
        u["mean load"] = self.trace.groupby("car")["load"].mean()
        return u

    def busiest_window(self, minutes: int = 15) -> float:
        """Clock hour at which the busiest `minutes`-long window starts."""
        per = self.by_period(f"{minutes}min")["arrivals"]
        ts = per.idxmax()
        return ts.hour + ts.minute / 60

    def export(self, outdir: Path) -> None:
        self.passengers.to_csv(outdir / "passengers.csv", index=False)
        self.trace.to_csv(outdir / "car_trace.csv", index=False)
        self.by_period().to_csv(outdir / "by_15min.csv")
        self.summary().to_csv(outdir / "summary.csv", header=["value"])


# --------------------------------------------------------------------------
# Plots (matplotlib)
# --------------------------------------------------------------------------
def floor_label(f: int) -> str:
    return "L" if f == 0 else str(f)


def plot_dashboard(res: Results, path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    cfg, pax = res.cfg, res.passengers
    hhmm = mdates.DateFormatter("%H:%M")
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    s = res.summary()
    fig.suptitle(
        f"Elevator simulation: {cfg.floors} floors, {cfg.elevators} cars x {cfg.capacity} pax   |   "
        f"{int(s['passengers'])} passengers, mean wait {s['mean wait (s)']:.0f} s, "
        f"95th pct {s['95th pct wait (s)']:.0f} s, mean journey {s['mean journey (s)']:.0f} s",
        fontsize=13, fontweight="bold")

    # (a) traffic and waiting time through the day
    ax = axes[0, 0]
    per = res.by_period("15min")
    ax.bar(per.index, per["arrivals"] / 15, width=13 / 1440, align="edge",
           color="0.8", label="arrivals / min")
    ax.set_ylabel("arrivals per minute")
    ax2 = ax.twinx()
    ax2.plot(per.index + pd.Timedelta("7.5min"), per["mean_wait"], color=colors[0], lw=2, label="mean wait")
    ax2.plot(per.index + pd.Timedelta("7.5min"), per["p95_wait"], color=colors[3], lw=1.5, ls="--",
             label="95th pct wait")
    ax2.set_ylabel("wait (s)")
    ax.xaxis.set_major_formatter(hhmm)
    ax.set_title("Traffic and waiting time (15-min bins)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    # (b) waiting-time distribution
    ax = axes[0, 1]
    w = pax["wait"].dropna()
    ax.hist(w, bins=np.arange(0, w.max() + 5, 5), color=colors[0], alpha=0.8)
    for val, name, c in [(w.mean(), "mean", "k"), (w.quantile(0.95), "95th pct", colors[3])]:
        ax.axvline(val, color=c, ls="--")
        ax.text(val, ax.get_ylim()[1] * 0.95, f" {name}\n {val:.0f} s", color=c, va="top", fontsize=9)
    ax.set_xlabel("wait for a car (s)")
    ax.set_ylabel("passengers")
    ax.set_title("Waiting-time distribution")

    # (c) space-time diagram of the busiest 15 minutes
    ax = axes[0, 2]
    start = res.busiest_window(15)
    t0 = int((start - cfg.start_hour) * 3600)
    win = res.trace[(res.trace["t"] >= t0) & (res.trace["t"] < t0 + 900)]
    for cid, g in win.groupby("car"):
        ax.plot(g["time"], g["position"], color=colors[cid % 10], lw=1.4, label=f"car {cid}")
    arr = pax[(pax["arrival"] >= t0) & (pax["arrival"] < t0 + 900)]
    for d, m in (("up", "^"), ("down", "v")):
        a = arr[arr["direction"] == d]
        ax.scatter(a["time"], a["origin"], marker=m, s=14, color="0.35", zorder=3,
                   label=f"call {d}")
    ax.set_yticks(range(cfg.floors), [floor_label(f) for f in range(cfg.floors)])
    ax.xaxis.set_major_formatter(hhmm)
    ax.set_ylabel("floor")
    ax.set_title("Car movements during the busiest 15 minutes")
    ax.legend(fontsize=8, ncol=2, loc="upper right")

    # (d) utilisation per car
    ax = axes[1, 0]
    u = res.utilization()
    left = np.zeros(len(u))
    for state, c in ((MOVING, colors[2]), (DOORS, colors[1]), (IDLE, "0.85")):
        ax.barh([f"car {i}" for i in u.index], u[state], left=left, color=c, label=state)
        left += u[state].to_numpy()
    for i, load in enumerate(u["mean load"]):
        ax.text(101, i, f"avg load {load:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, 125)
    ax.set_xlabel("% of time")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
    ax.set_title("Car utilisation")

    # (e) mean wait by origin floor and direction
    ax = axes[1, 1]
    by_floor = pax.pivot_table(index="origin", columns="direction", values="wait", aggfunc="mean")
    by_floor = by_floor.reindex(columns=["up", "down"])
    y = np.arange(len(by_floor))
    ax.barh(y - 0.2, by_floor["up"], height=0.4, color=colors[0], label="going up")
    ax.barh(y + 0.2, by_floor["down"], height=0.4, color=colors[1], label="going down")
    ax.set_yticks(y, [floor_label(f) for f in by_floor.index])
    ax.set_xlabel("mean wait (s)")
    ax.set_ylabel("origin floor")
    ax.legend(fontsize=9)
    ax.set_title("Mean wait by floor")

    # (f) people waiting vs. riding
    ax = axes[1, 2]
    waiting = res.queue.set_index("time")["waiting"].resample("1min").mean()
    riding = res.trace.groupby("time")["load"].sum().resample("1min").mean()
    ax.fill_between(riding.index, riding, step="mid", alpha=0.35, color=colors[2], label="riding")
    ax.plot(waiting.index, waiting, color=colors[3], lw=1.5, label="waiting in lobbies")
    ax.xaxis.set_major_formatter(hhmm)
    ax.set_ylabel("people (1-min average)")
    ax.legend(fontsize=9)
    ax.set_title("People waiting vs. riding")

    fig.savefig(path, dpi=120)


def plot_comparison(table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    n = table.index
    a1.plot(n, table["mean wait (s)"], "o-", label="mean wait")
    a1.plot(n, table["95th pct wait (s)"], "s--", label="95th pct wait")
    a1.plot(n, table["mean journey (s)"], "^-", label="mean journey")
    a1.set_xlabel("number of elevators")
    a1.set_ylabel("seconds")
    a1.set_xticks(n)
    a1.grid(alpha=0.3)
    a1.legend()
    a1.set_title("Service level vs. fleet size")
    a2.bar(n, table["car busy (%)"], color="tab:green")
    a2.set_xlabel("number of elevators")
    a2.set_ylabel("% of time not idle")
    a2.set_xticks(n)
    a2.set_title("Average car utilisation")
    fig.savefig(path, dpi=120)


def animate(res: Results, start_hour: float, minutes: float, speed: int,
            save: Path | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    cfg, pax = res.cfg, res.passengers
    t0 = int((start_hour - cfg.start_hour) * 3600)
    t1 = min(t0 + int(minutes * 60), int(res.trace["t"].max()) + 1)
    win = res.trace[(res.trace["t"] >= t0) & (res.trace["t"] < t1)]
    pos = win.pivot(index="t", columns="car", values="position")
    load = win.pivot(index="t", columns="car", values="load")

    arrival = pax["arrival"].to_numpy()
    board = pax["board"].fillna(np.inf).to_numpy()
    origin = pax["origin"].to_numpy()
    up = (pax["direction"] == "up").to_numpy()

    fig, ax = plt.subplots(figsize=(4 + 1.1 * cfg.elevators, 0.55 * cfg.floors + 2))
    ax.set_xlim(-2.6, cfg.elevators - 0.3)
    ax.set_ylim(-0.6, cfg.floors - 0.4)
    ax.set_xticks([])
    ax.set_yticks(range(cfg.floors), [floor_label(f) for f in range(cfg.floors)])
    for f in range(cfg.floors + 1):
        ax.axhline(f - 0.5, color="0.85", lw=0.8)
    for c in range(cfg.elevators):
        ax.axvspan(c - 0.4, c + 0.4, color="0.95", zorder=0)
    ax.axvline(-0.7, color="0.5")
    ax.text(-1.65, cfg.floors - 0.45, "waiting  ▲ / ▼", ha="center", va="bottom", fontsize=9)

    cmap = plt.cm.RdYlGn_r
    rects, labels = [], []
    for c in range(cfg.elevators):
        r = Rectangle((c - 0.35, 0), 0.7, 0.8, ec="k", lw=1)
        ax.add_patch(r)
        rects.append(r)
        labels.append(ax.text(c, 0, "", ha="center", va="center", fontsize=9, fontweight="bold"))
    wait_text = [ax.text(-1.65, f, "", ha="center", va="center", fontsize=10) for f in range(cfg.floors)]
    title = ax.set_title("", pad=22)

    def update(t):
        for c in range(cfg.elevators):
            y, n = pos.at[t, c], load.at[t, c]
            rects[c].set_y(y - 0.4)
            rects[c].set_facecolor(cmap(0.15 + 0.85 * n / cfg.capacity))
            labels[c].set_position((c, y))
            labels[c].set_text(f"{n}")
        waiting = (arrival <= t) & (board > t)
        ups = np.bincount(origin[waiting & up], minlength=cfg.floors)
        downs = np.bincount(origin[waiting & ~up], minlength=cfg.floors)
        for f in range(cfg.floors):
            wait_text[f].set_text(f"{ups[f] or '·'}  /  {downs[f] or '·'}")
        clock = BASE_DATE + pd.to_timedelta(cfg.start_hour, unit="h") + pd.to_timedelta(t, unit="s")
        title.set_text(f"{clock:%H:%M:%S}   (number in car = passengers aboard)")
        return rects + labels + wait_text + [title]

    frames = range(t0, t1, max(1, speed))
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    if save:
        anim.save(save, writer=PillowWriter(fps=20))
        print(f"saved animation -> {save}")
    else:
        plt.show()
    return anim


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floors", type=int, default=12)
    ap.add_argument("--elevators", type=int, default=4)
    ap.add_argument("--capacity", type=int, default=10)
    ap.add_argument("--hours", type=float, default=12.0, help="simulated hours from --start-hour")
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--traffic", type=float, default=1.0, help="traffic multiplier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent / "output")
    ap.add_argument("--compare", action="store_true", help="also sweep 2..7 elevators")
    ap.add_argument("--animate", action="store_true", help="animate the busiest 15 minutes")
    ap.add_argument("--anim-start", type=float, help="clock hour to start the animation (e.g. 8.5)")
    ap.add_argument("--anim-minutes", type=float, default=15)
    ap.add_argument("--anim-speed", type=int, default=4, help="simulated seconds per frame")
    ap.add_argument("--save-anim", type=Path, help="write the animation to a .gif instead of showing it")
    ap.add_argument("--no-show", action="store_true", help="don't open plot windows")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config(floors=args.floors, elevators=args.elevators, capacity=args.capacity,
                 hours=args.hours, start_hour=args.start_hour,
                 traffic_scale=args.traffic, seed=args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)

    print(f"Simulating {cfg.floors} floors, {cfg.elevators} elevators, "
          f"{cfg.hours:g} h from {cfg.start_hour:05.2f} ...")
    res = Simulation(cfg).run()

    print("\n=== Summary ===")
    print(res.summary().round(1).to_string())
    print("\n=== By hour ===")
    print(res.by_period("1h").round(1).to_string())
    print("\n=== Car utilisation (% of time) ===")
    print(res.utilization().round(1).to_string())

    res.export(args.outdir)
    plot_dashboard(res, args.outdir / "dashboard.png")
    print(f"\nwrote CSVs and dashboard.png -> {args.outdir}")

    if args.compare:
        rows = {}
        for n in range(2, 8):
            r = Simulation(replace(cfg, elevators=n)).run()
            rows[n] = r.summary()
            print(f"  {n} cars: mean wait {rows[n]['mean wait (s)']:.1f} s")
        table = pd.DataFrame(rows).T.rename_axis("elevators")
        print("\n=== Fleet-size comparison ===")
        print(table.round(1).to_string())
        table.to_csv(args.outdir / "comparison.csv")
        plot_comparison(table, args.outdir / "comparison.png")

    if args.animate or args.save_anim:
        start = args.anim_start if args.anim_start is not None else res.busiest_window(15)
        animate(res, start, args.anim_minutes, args.anim_speed, args.save_anim)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
