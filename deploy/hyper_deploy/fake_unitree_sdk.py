"""Stand-in for unitree_actuator_sdk (same names as the official python binding) for dry runs and tests without
motors. Each motor is a rotor-side point inertia driven by the FOC PD law the real motor runs:
    tau_rotor = tau + kp (q - q_r) + kd (dq - dq_r),  output torque = tau_rotor * gear.
"""
import enum
import threading

GEAR = 6.33


class MotorType(enum.Enum):
    A1 = 0
    B1 = 1
    GO_M8010_6 = 2


class MotorMode(enum.Enum):
    BRAKE = 0
    FOC = 1
    CALIBRATE = 2


def queryMotorMode(motortype, motormode):
    return int(motormode.value)


def queryGearRatio(motortype):
    return GEAR if motortype == MotorType.GO_M8010_6 else 9.1


class MotorCmd:
    def __init__(self):
        self.motorType = MotorType.GO_M8010_6
        self.id = 0
        self.mode = 0
        self.tau = self.dq = self.q = self.kp = self.kd = 0.0


class MotorData:
    def __init__(self):
        self.motorType = MotorType.GO_M8010_6
        self.motor_id = 0
        self.mode = 0
        self.temp = 30
        self.merror = 0
        self.tau = self.dq = self.q = 0.0
        self.correct = False


class SerialPort:
    """Motors keyed by id; `initial_rotor_q` lets tests start motors at chosen rotor angles."""

    registry = {}

    def __init__(self, port, initial_rotor_q=None):
        self.port = port
        self.motors = {}
        self.lock = threading.Lock()
        self.initial = dict(initial_rotor_q or SerialPort.registry.get(port, {}))
        self.log = []

    def sendRecv(self, cmd, data, dt=0.002, rotor_inertia_out=0.05):
        with self.lock:
            m = self.motors.setdefault(cmd.id, {"q": float(self.initial.get(cmd.id, 0.0)), "dq": 0.0})
            if cmd.mode == queryMotorMode(MotorType.GO_M8010_6, MotorMode.FOC):
                tau_r = cmd.tau + cmd.kp * (cmd.q - m["q"]) + cmd.kd * (cmd.dq - m["dq"])
            else:
                tau_r = 0.0
            # integrate in output space: I_out ddq_out = tau_out
            tau_out = tau_r * GEAR
            dq_out = m["dq"] / GEAR + tau_out / rotor_inertia_out * dt
            m["dq"] = dq_out * GEAR
            m["q"] += m["dq"] * dt
            self.log.append((cmd.id, cmd.q, cmd.kp, cmd.kd, cmd.tau))
            data.motor_id = cmd.id
            data.q, data.dq, data.tau = m["q"], m["dq"], tau_r
            data.temp, data.merror, data.correct = 30, 0, True
        return True
