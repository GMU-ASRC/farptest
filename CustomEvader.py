from swarmsim.agent.control.WaypointPIDController import WaypointPIDController, DEFAULT_PID_ARGS
from swarmsim.sensors.BinaryFOVSensor import BinaryFOVSensor


class CustomEvader(WaypointPIDController):
    def __init__(
        self,
        agent=None, parent=None,
        targeter_sensor_id=None,
        speed_pid=DEFAULT_PID_ARGS,
        steer_pid=DEFAULT_PID_ARGS,
        static_waypoint=None,
        **kwargs
    ):
        super().__init__(agent=agent, parent=parent,
                         sensor_id=targeter_sensor_id,
                         speed_pid=speed_pid, steer_pid=steer_pid,
                         static_waypoint=static_waypoint,
                         **kwargs)

    def get_actions(self, agent):
        binary: BinaryFOVSensor = agent.sensors[1]
        v, w = (0, 0) if binary.current_state else super().get_actions(agent)
        return v, w
