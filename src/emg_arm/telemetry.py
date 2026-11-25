"""Timing of the whole pipeline: armband signal -> model -> gesture decision -> command -> USB packet to the xArm.

Pure Python (no Qt). All times are ``time.perf_counter()`` seconds taken on this computer.

For one gesture that moves the arm, the stages add up to the total exactly::

    signal arrives -> [wait] -> [preprocess + classify] -> [deliver] -> [vote] -> [dispatch] -> [USB send] -> done

* wait      new samples arrived from the armband, until the model starts on the window that contains them
* deliver   model finished, until the control logic received the result
* vote      first window that voted for the gesture, until the gesture counted as stable
* dispatch  stable gesture, until the command was issued (one-shot gestures include their hold time)
* USB send  command issued, until the packet had been written to the xArm

Not included, because this computer cannot measure them: the radio/cable delay from the armband (its clock is not
synchronised with ours) and the time the servos take to physically move.
"""

from __future__ import annotations

import csv
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

KEEP = 3000                      # newest records kept of each kind
GIVE_UP_S = 10.0                 # a gesture that caused no command for this long is closed as "not sent"


def _ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else (b - a) * 1000.0


@dataclass
class Timing:
    """One signal window on its way through the model."""
    signal: float                          # the newest samples of the window arrived from the armband
    start: float                           # the model started on it
    prepared: Optional[float]              # filtering + features + scaling finished (None: model does not say)
    done: float                            # classifier finished
    since_previous: Optional[float] = None  # seconds of new signal collected since the previous window
    delivered: float = 0.0                 # result reached the control logic

    @property
    def wait_ms(self):
        return _ms(self.signal, self.start)

    @property
    def preprocess_ms(self):
        return _ms(self.start, self.prepared)

    @property
    def classify_ms(self):
        return _ms(self.prepared, self.done)

    @property
    def inference_ms(self):
        return _ms(self.start, self.done)

    @property
    def deliver_ms(self):
        return _ms(self.done, self.delivered)

    @property
    def total_ms(self):
        return _ms(self.signal, self.delivered)


@dataclass
class Prediction:
    wall: float
    label: str
    confidence: float
    gesture: Optional[str]                 # accepted gesture id (None = ignored)
    timing: Timing


@dataclass
class GestureEvent:
    """One activation of a gesture, followed until its first packet reached the arm."""
    wall: float
    gesture: str
    first: Prediction                      # the first window that voted for it
    stable: float
    command: Optional[float] = None
    sent: Optional[float] = None
    status: str = "waiting"
    data: str = ""
    servos: tuple = ()

    @property
    def vote_ms(self):
        return _ms(self.first.timing.delivered, self.stable)

    @property
    def dispatch_ms(self):
        return _ms(self.stable, self.command)

    @property
    def send_ms(self):
        return None if self.sent is None else _ms(self.command, self.sent)

    @property
    def decision_ms(self):
        """Signal arrival -> stable gesture."""
        return _ms(self.first.timing.signal, self.stable)

    @property
    def total_ms(self):
        """Signal arrival -> packet written to the xArm."""
        return _ms(self.first.timing.signal, self.sent)


@dataclass
class Packet:
    """One command written to the arm."""
    wall: float
    start: float
    end: float
    targets: Dict[int, int]
    duration_ms: int
    data: str
    cause: str = ""

    @property
    def write_ms(self):
        return _ms(self.start, self.end)


class PipelineLog:
    def __init__(self, vote_window: int = 5, names: Optional[Dict[int, str]] = None):
        self.names = dict(names or {})
        self.cause = ""                    # why the arm is being commanded right now (set by the control logic)
        self._lock = threading.Lock()      # packets arrive from the motion thread, everything else from the GUI
        self._votes: deque = deque(maxlen=max(1, int(vote_window)))
        self.clear()

    def clear(self):
        with self._lock:
            self.predictions: deque = deque(maxlen=KEEP)
            self.gestures: deque = deque(maxlen=KEEP)
            self.packets: deque = deque(maxlen=KEEP)
            self._open: Dict[str, GestureEvent] = {}
            self._votes.clear()
            self.versions = {"predictions": 0, "gestures": 0, "packets": 0}

    def _bump(self, kind: str):
        self.versions[kind] += 1

    # ------------------------------------------------------------------ model side
    def forget_votes(self):
        """The gesture filter was reset: earlier windows no longer count towards a decision."""
        with self._lock:
            self._votes.clear()

    def add_prediction(self, label: str, confidence: float, gesture: Optional[str], timing: Timing) -> Prediction:
        p = Prediction(time.time(), label, float(confidence), gesture, timing)
        with self._lock:
            self.predictions.append(p)
            self._votes.append(p)
            self._bump("predictions")
        return p

    def gesture_stable(self, gesture: str, now: float, closed_because: str = "") -> Optional[GestureEvent]:
        """A gesture just became the stable decision. ``closed_because``: why no command will follow."""
        with self._lock:
            first = next((p for p in self._votes if p.gesture == gesture), None)
            if first is None:
                return None
            event = GestureEvent(time.time(), gesture, first, now)
            old = self._open.pop(gesture, None)
            if old is not None and old.command is None:
                old.status = "not sent"
            if closed_because:
                event.status = closed_because
            else:
                self._open[gesture] = event
            self.gestures.append(event)
            self._bump("gestures")
            return event

    # ------------------------------------------------------------------ arm side
    def commanded(self, gesture: str, servos: Iterable[int], at: float, note: str = ""):
        """The control logic issued a command for this gesture. No servos = nothing is sent to the arm."""
        with self._lock:
            event = self._open.get(gesture)
            if event is None or event.command is not None:
                return
            event.command, event.servos = at, tuple(int(s) for s in servos)
            if not event.servos:
                event.status, event.data = "done (no arm packet)", note
                self._open.pop(gesture, None)
            else:
                for packet in self.packets:            # one-shot actions write their packet before we get here
                    if packet.start >= at and set(packet.targets) & set(event.servos):
                        self._complete(event, packet)
                        break
            self._bump("gestures")

    def packet_sent(self, targets: Dict[int, int], duration_ms: int, start: float, end: float):
        data = ", ".join(f"{self.names.get(s, f'servo {s}')} (servo {s}) → {p}" for s, p in sorted(targets.items()))
        packet = Packet(time.time(), start, end, dict(targets), int(duration_ms), data, self.cause)
        with self._lock:
            self.packets.append(packet)
            self._bump("packets")
            for event in list(self._open.values()):
                if event.command is not None and start >= event.command and set(targets) & set(event.servos):
                    self._complete(event, packet)
                    self._bump("gestures")

    def _complete(self, event: GestureEvent, packet: Packet):
        event.sent, event.status = packet.end, "sent"
        event.data = f"{packet.data}  in {packet.duration_ms} ms"
        packet.cause = f"gesture: {event.gesture}"
        self._open.pop(event.gesture, None)

    def close_open(self, reason: str):
        """E.g. disarmed: gestures still waiting for a command will not get one. Commands already issued
        keep waiting for their packet (the "go home" gesture disarms before its packet is written)."""
        with self._lock:
            waiting = [g for g, e in self._open.items() if e.command is None]
            for gesture in waiting:
                self._open.pop(gesture).status = reason
            if waiting:
                self._bump("gestures")

    def expire(self, now: float):
        with self._lock:
            stale = [g for g, e in self._open.items() if now - e.stable > GIVE_UP_S]
            for g in stale:
                self._open.pop(g).status = "not sent"
            if stale:
                self._bump("gestures")

    # ------------------------------------------------------------------ reading
    def snapshot(self, kind: str, limit: Optional[int] = None) -> list:
        """Newest first."""
        with self._lock:
            items = list(getattr(self, kind))
        items.reverse()
        return items[:limit] if limit else items

    def per_gesture(self) -> List[dict]:
        """Averages for each predicted label, and for each gesture that reached the arm."""
        rows: Dict[str, dict] = {}

        def row(key):
            return rows.setdefault(key, {"gesture": key, "windows": 0, "pre": [], "cls": [], "inf": [], "result": [],
                                         "decision": [], "arm": []})

        for p in self.snapshot("predictions"):
            r = row(p.gesture or p.label)
            r["windows"] += 1
            for key, value in (("pre", p.timing.preprocess_ms), ("cls", p.timing.classify_ms),
                               ("inf", p.timing.inference_ms), ("result", p.timing.total_ms)):
                if value is not None:
                    r[key].append(value)
        for e in self.snapshot("gestures"):
            r = row(e.gesture)
            r["decision"].append(e.decision_ms)
            if e.total_ms is not None:
                r["arm"].append(e.total_ms)
        out = []
        for r in sorted(rows.values(), key=lambda r: r["gesture"]):
            avg = {k: (sum(v) / len(v) if v else None) for k, v in r.items() if isinstance(v, list)}
            out.append({"gesture": r["gesture"], "windows": r["windows"], "sent": len(r["arm"]),
                        "inf_max": max(r["inf"]) if r["inf"] else None, **avg})
        return out

    def export_csv(self, path) -> int:
        """Every record that is still kept, oldest first. Returns the number of rows."""
        fields = ["kind", "clock_time", "gesture", "label", "confidence", "new_signal_ms", "wait_ms", "preprocess_ms",
                  "classify_ms", "inference_ms", "deliver_ms", "signal_to_result_ms", "vote_ms", "dispatch_ms",
                  "usb_send_ms", "signal_to_decision_ms", "signal_to_arm_ms", "duration_ms", "status", "data"]

        def clock(t):
            return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"

        def timing(t: Timing):
            return {"new_signal_ms": None if t.since_previous is None else t.since_previous * 1000.0,
                    "wait_ms": t.wait_ms, "preprocess_ms": t.preprocess_ms, "classify_ms": t.classify_ms,
                    "inference_ms": t.inference_ms, "deliver_ms": t.deliver_ms, "signal_to_result_ms": t.total_ms}

        rows = []
        for p in self.snapshot("predictions"):
            rows.append((p.wall, {"kind": "prediction", "gesture": p.gesture or "", "label": p.label,
                                  "confidence": p.confidence, **timing(p.timing)}))
        for e in self.snapshot("gestures"):
            rows.append((e.wall, {"kind": "gesture", "gesture": e.gesture, "label": e.first.label,
                                  "confidence": e.first.confidence, **timing(e.first.timing), "vote_ms": e.vote_ms,
                                  "dispatch_ms": e.dispatch_ms, "usb_send_ms": e.send_ms,
                                  "signal_to_decision_ms": e.decision_ms, "signal_to_arm_ms": e.total_ms,
                                  "status": e.status, "data": e.data}))
        for k in self.snapshot("packets"):
            rows.append((k.wall, {"kind": "xarm_packet", "usb_send_ms": k.write_ms, "duration_ms": k.duration_ms,
                                  "status": k.cause, "data": k.data}))
        rows.sort(key=lambda r: r[0])
        with open(path, "w", newline="", encoding="utf-8-sig") as f:      # -sig: Excel reads the arrows correctly
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for wall, row in rows:
                row = {k: (f"{v:.3f}" if isinstance(v, float) else "" if v is None else v) for k, v in row.items()}
                writer.writerow({"clock_time": clock(wall), **row})
        return len(rows)
