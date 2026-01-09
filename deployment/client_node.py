import time
import torch
import math
import threading
import numpy as np
from typing import Any, Union, Dict

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__HandState_
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
from dataclasses import dataclass
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_ as LowCmdHG,
    HandCmd_ as HandCmd,
    LowState_ as LowStateHG,
    HandState_)

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_dex3, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from gr00t.eval.service import ExternalRobotInferenceClient
from common.remote_controller import RemoteController, KeyMap
from robotconfig import RobotConfig
from realsense_camera import RealSenseCamera
from camera_client import HeadCameraClient, CameraConfig

@dataclass
class ControlConfig:
    control_dt: float = 0.02
    arm_kp_multiplier: float = 0.3
    arm_kd_multiplier: float = 1.2
    hand_kp: float = 1.5
    hand_kd: float = 0.1
    move_to_default_time: float = 2.0
    save_image_interval: int = 2
    # Safety parameters from reference code
    kp_high: float = 300.0
    kd_high: float = 3.0
    kp_low: float = 80.0
    kd_low: float = 3.0
    kp_wrist: float = 40.0
    kd_wrist: float = 1.5
    arm_velocity_limit: float = 20.0
    max_arm_velocity: float = 30.0

@dataclass
class PolicyConfig:
    host: str = "192.168.123.103"
    port: int = 5000


class Controller:
    def __init__(self, config: RobotConfig, sim=False) -> None:
        self.simulation = sim
        self.config = config
        self.control_config = ControlConfig()
        self.policy_config = PolicyConfig()

        # Add holding state variables for threading
        self.holding_position = False
        self.last_arm_positions = None
        self.last_hand_positions = None
        self.hold_thread = None
        self.stop_holding = threading.Event()
        self.position_lock = threading.Lock()  # Thread safety for position updates

        if self.simulation:
            self.policy_config.host = "0.0.0.0"
            print("In Simulation")
            self.camera_config = CameraConfig()
            self.sim_camera = HeadCameraClient(self.camera_config)
        else:
            print("Running on physical robot")
            self.remote_controller = RemoteController()
            self.camera = RealSenseCamera(img_shape=config.camera_image_shape, fps=config.camera_fps, serial_number=config.camera_id)

        self.policy = ExternalRobotInferenceClient(
            host=self.policy_config.host,
            port=self.policy_config.port
        )

        if config.msg_type == "hg":
            # Initialise commands
            self.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.hand_state = unitree_hg_msg_dds__HandState_()
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            # Initialise body publishers and subscribers
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

            # Initialise hand publishers and subscribers
            self.lefthand_publisher = ChannelPublisher(config.dex3_leftcmd_topic, HandCmd)
            self.lefthand_publisher.Init()

            self.righthand_publisher = ChannelPublisher(config.dex3_rightcmd_topic, HandCmd)
            self.righthand_publisher.Init()

            self.lefthand_subscriber = ChannelSubscriber(config.dex3_leftstate_topic, HandState_)
            self.lefthand_subscriber.Init()

            self.righthand_subscriber = ChannelSubscriber(config.dex3_rightstate_topic, HandState_)
            self.righthand_subscriber.Init()

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()
        self.wait_for_hand_state()
        print("initialised")

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
            init_cmd_dex3(self.left_hand_cmd)
            init_cmd_dex3(self.right_hand_cmd)

    def _holding_loop(self):
        """Background thread to continuously send last known positions"""
        while not self.stop_holding.is_set():
            with self.position_lock:
                if self.holding_position and self.last_arm_positions is not None:
                    # Set arm commands with last known positions
                    self._set_arm_commands(self.last_arm_positions[0], self.last_arm_positions[1])
                    self._set_hand_commands(self.last_hand_positions[0], self.last_hand_positions[1])

                    # Send commands
                    self.send_cmd(self.low_cmd)
                    self.lefthand_publisher.Write(self.left_hand_cmd)
                    self.righthand_publisher.Write(self.right_hand_cmd)

            time.sleep(self.config.control_dt)

    def start_holding(self, last_left_arm, last_right_arm, last_left_hand, last_right_hand):
        """Start holding the last positions"""
        with self.position_lock:
            self.last_arm_positions = (last_left_arm, last_right_arm)
            self.last_hand_positions = (last_left_hand, last_right_hand)
            self.holding_position = True

        if self.hold_thread is None or not self.hold_thread.is_alive():
            self.stop_holding.clear()
            self.hold_thread = threading.Thread(target=self._holding_loop)
            self.hold_thread.daemon = True
            self.hold_thread.start()

    def stop_holding_position(self):
        """Stop the holding loop"""
        with self.position_lock:
            self.holding_position = False

    def cleanup_threads(self):
        """Clean up threads on exit"""
        self.stop_holding.set()
        if self.hold_thread and self.hold_thread.is_alive():
            self.hold_thread.join(timeout=1.0)

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        if not self.simulation:
            self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def wait_for_hand_state(self):
        while self.hand_state.motor_state is None:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the hands.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.right_hand_cmd)
            create_zero_cmd(self.left_hand_cmd)
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self.lefthand_publisher.Write(self.left_hand_cmd)
            self.righthand_publisher.Write(self.right_hand_cmd)
            left_hand_state, right_hand_state = self._read_hand_state()
            print("LEFT_HAND_OBS", left_hand_state)
            print("RIGHT_HAND_OBS", right_hand_state)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_hands_to_default(self):
        """Send both hands to default position"""
        # Set hands to neutral/open position
        for i in range(7):  # Assuming 7 DOF hands
            # Left hand
            ris_mode = self._RIS_Mode(id = i, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_hand_cmd.motor_cmd[i].mode = motor_mode

            if i == 0:
                self.right_hand_cmd.motor_cmd[i].q = 0.0 #- (math.pi / 8) # Neutral position
            else:
                self.right_hand_cmd.motor_cmd[i].q = 0.0  # Neutral position
            self.left_hand_cmd.motor_cmd[i].qd = 0
            self.left_hand_cmd.motor_cmd[i].kp = self.config.hand_kps
            self.left_hand_cmd.motor_cmd[i].kd = self.config.hand_kds
            self.left_hand_cmd.motor_cmd[i].tau = 0

            # Right hand
            ris_mode = self._RIS_Mode(id = i, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_hand_cmd.motor_cmd[i].mode = motor_mode

            if i == 0:
                self.right_hand_cmd.motor_cmd[i].q = 0.0 #- (math.pi / 8) # Neutral position
            else:
                self.right_hand_cmd.motor_cmd[i].q = 0.0  # Neutral position
            self.right_hand_cmd.motor_cmd[i].qd = 0
            self.right_hand_cmd.motor_cmd[i].kp = self.config.hand_kps
            self.right_hand_cmd.motor_cmd[i].kd = self.config.hand_kds
            self.right_hand_cmd.motor_cmd[i].tau = 0

        # Send hand commands
        self.lefthand_publisher.Write(self.left_hand_cmd)
        self.righthand_publisher.Write(self.right_hand_cmd)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _read_arm_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Read arm joint states"""
        # G1_29 arm indices: 15-21 (left), 22-28 (right) - 7 joints per arm
        left_arm_q = np.array([self.low_state.motor_state[i].q for i in range(15, 22)])
        right_arm_q = np.array([self.low_state.motor_state[i].q for i in range(22, 29)])

        if len(left_arm_q) == 7 and len(right_arm_q) == 7:
            return (np.expand_dims(left_arm_q, axis=0),
                   np.expand_dims(right_arm_q, axis=0))
        else:
            raise Exception("Could not read arm state")

    def _read_hand_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Read hand joint states with fallback to random"""
        left_hand_data = self.lefthand_subscriber.Read()
        right_hand_data = self.righthand_subscriber.Read()

        if left_hand_data is not None and right_hand_data is not None:
            left_hand_q = np.array([left_hand_data.motor_state[i].q for i in range(7)])
            right_hand_q = np.array([right_hand_data.motor_state[i].q for i in range(7)])

            return (np.expand_dims(left_hand_q, axis=0),
                   np.expand_dims(right_hand_q, axis=0))
        else:
            raise Exception("Could not read arm state")

    #def _read_hand_state(self) -> tuple[np.ndarray, np.ndarray]:
    #    """Read hand joint states and reorder to match model expectations"""
    #    left_hand_data = self.lefthand_subscriber.Read()
    #    right_hand_data = self.righthand_subscriber.Read()

    #    if left_hand_data is not None and right_hand_data is not None:
    #        # Read raw joint positions
    #        left_hand_raw = np.array([left_hand_data.motor_state[i].q for i in range(7)])
    #        right_hand_raw = np.array([right_hand_data.motor_state[i].q for i in range(7)])
    #
    #        # Robot hand joint order: [0, 1, 2, 5, 6, 3, 4]
    #        # Model expects:          [0, 1, 2, 3, 4, 5, 6] (normal order)
    #
    #        # Create inverse mapping: where to get each model joint from robot joints
    #        robot_order = [0, 1, 2, 5, 6, 3, 4]  # Robot's physical order
    #        model_to_robot_mapping = [0] * 7
    #        for model_idx, robot_idx in enumerate(robot_order):
    #            model_to_robot_mapping[robot_idx] = model_idx
    #        # Reorder joints to match model expectations
    #        left_hand_q = np.array([left_hand_raw[model_to_robot_mapping[i]] for i in range(7)])
    #        right_hand_q = np.array([right_hand_raw[model_to_robot_mapping[i]] for i in range(7)])

    #        return (np.expand_dims(left_hand_q, axis=0),
    #               np.expand_dims(right_hand_q, axis=0))
    #    else:
    #        raise Exception("Could not read hand state")

    def _build_observation(self, camera_image: np.ndarray,
                          left_arm_state: np.ndarray, right_arm_state: np.ndarray,
                          left_hand_state: np.ndarray, right_hand_state: np.ndarray) -> Dict[str, Any]:
        """Build observation dictionary for policy"""

        "Remapping"
        print("SIZE: ", left_hand_state.shape)
        left_hand_state[0][0] = left_hand_state[0][0] * -1
        right_hand_state[0][0] = right_hand_state[0][0] * -1
        left_hand_state[0][1] = left_hand_state[0][1] * -1
        right_hand_state[0][1] = right_hand_state[0][1] * -1
        left_hand_state[0][2] = left_hand_state[0][2] * -1
        right_hand_state[0][2] = right_hand_state[0][2] * -1

        return {
            #"video.cam_right_high": camera_image,
            "video.rs_view": camera_image,
            "state.left_arm": left_arm_state,
            "state.right_arm": right_arm_state,
            "state.left_hand": left_hand_state,
            "state.right_hand": right_hand_state,
            "annotation.human.task_description": ["Pick up the red apple and place it on the plate"]
            # "annotation.human.task_description": [
            #     #"Stack the three cubic blocks on the desktop from bottom to top in the order of red, white, and black on the yellow tape affixed to the desktop."
            #     #"Stack the three cubic blocks on the desktop from bottom to top in the order of red, white, and black on the desktop."
            #     "place the red cube on the green plate"

            # ],
        }

    def _set_arm_commands(self, left_arm_action: np.ndarray, right_arm_action: np.ndarray):
        """Set arm motor commands"""
        # Left arm (G1_29 indices: 15-21)
        left_arm_indices = [15, 16, 17, 18, 19, 20, 21]
        for i, motor_idx in enumerate(left_arm_indices):
            self.low_cmd.motor_cmd[motor_idx].q = left_arm_action[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i] * self.control_config.arm_kp_multiplier
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i] * self.control_config.arm_kd_multiplier
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # Right arm (G1_29 indices: 22-28)
        right_arm_indices = [22, 23, 24, 25, 26, 27, 28]
        for i, motor_idx in enumerate(right_arm_indices):
            self.low_cmd.motor_cmd[motor_idx].q = right_arm_action[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i] * self.control_config.arm_kp_multiplier
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i] * self.control_config.arm_kd_multiplier
            self.low_cmd.motor_cmd[motor_idx].tau = 0

    def _set_hand_commands(self, left_hand_action: np.ndarray, right_hand_action: np.ndarray):
        """Set hand motor commands"""
        for i, motor_idx in enumerate(self.config.left_hand):
            ris_mode = self._RIS_Mode(id = motor_idx, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_hand_cmd.motor_cmd[motor_idx].mode = motor_mode

            if motor_idx == 0:
                self.left_hand_cmd.motor_cmd[motor_idx].q = (-1 * left_hand_action[i]) #+ (math.pi / 8)
            elif motor_idx == 1 or motor_idx == 2:
                self.left_hand_cmd.motor_cmd[motor_idx].q = (-1 * left_hand_action[i])
            elif motor_idx == 5 or motor_idx == 6:
                self.left_hand_cmd.motor_cmd[motor_idx].q = (-1 * left_hand_action[i])
            else:
                self.left_hand_cmd.motor_cmd[motor_idx].q = left_hand_action[i]  # Neutral position

            #self.left_hand_cmd.motor_cmd[motor_idx].q = left_hand_action[i]  # Neutral position
            self.left_hand_cmd.motor_cmd[motor_idx].qd = 0
            self.left_hand_cmd.motor_cmd[motor_idx].kp = self.config.hand_kps #* 0.3
            self.left_hand_cmd.motor_cmd[motor_idx].kd = self.config.hand_kds #* 0.3
            self.left_hand_cmd.motor_cmd[motor_idx].tau = 0

        # Right hand
        for i, motor_idx in enumerate(self.config.right_hand):
            ris_mode = self._RIS_Mode(id = motor_idx, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_hand_cmd.motor_cmd[motor_idx].mode = motor_mode

            if motor_idx == 0:
                self.right_hand_cmd.motor_cmd[motor_idx].q = -1 * right_hand_action[i] #+ (math.pi / 4) # Neutral position
            elif motor_idx == 1 or motor_idx == 2:
                self.right_hand_cmd.motor_cmd[motor_idx].q = -1 * right_hand_action[i] #+ (math.pi / 8) # Neutral position
            else:
                self.right_hand_cmd.motor_cmd[motor_idx].q = right_hand_action[i]  # Neutral position

            #self.right_hand_cmd.motor_cmd[motor_idx].q = right_hand_action[i]  # Neutral position
            self.right_hand_cmd.motor_cmd[motor_idx].qd = 0
            self.right_hand_cmd.motor_cmd[motor_idx].kp = self.config.hand_kps #* 0.3
            self.right_hand_cmd.motor_cmd[motor_idx].kd = self.config.hand_kds #* 0.3
            self.right_hand_cmd.motor_cmd[motor_idx].tau = 0

    def _interpolate_to_first_action(self, action_chunk: Dict[str, Any], interpolation_steps: int = 3):
        """Smoothly interpolate from held position to first action of new chunk"""
        if self.last_arm_positions is None:
            return  # No previous position to interpolate from

        left_arm_actions = action_chunk['action.left_arm']
        right_arm_actions = action_chunk['action.right_arm']
        left_hand_actions = action_chunk['action.left_hand']
        right_hand_actions = action_chunk['action.right_hand']

        # Get target (first action of new chunk)
        target_combined_arm = np.concatenate([left_arm_actions[0], right_arm_actions[0]])
        target_left_arm = target_combined_arm[:7]
        target_right_arm = target_combined_arm[7:]
        target_left_hand = left_hand_actions[0]
        target_right_hand = right_hand_actions[0]

        # Get current held positions
        current_left_arm = self.last_arm_positions[0]
        current_right_arm = self.last_arm_positions[1]
        current_left_hand = self.last_hand_positions[0]
        current_right_hand = self.last_hand_positions[1]

        # Interpolate over several steps
        for step in range(1, interpolation_steps + 1):
            alpha = step / interpolation_steps

            # Interpolate arm positions
            interp_left_arm = current_left_arm + (target_left_arm - current_left_arm) * alpha
            interp_right_arm = current_right_arm + (target_right_arm - current_right_arm) * alpha

            # Interpolate hand positions
            interp_left_hand = current_left_hand + (target_left_hand - current_left_hand) * alpha
            interp_right_hand = current_right_hand + (target_right_hand - current_right_hand) * alpha

            # Set and send interpolated commands
            self._set_arm_commands(interp_left_arm, interp_right_arm)
            self._set_hand_commands(interp_left_hand, interp_right_hand)

            self.send_cmd(self.low_cmd)
            self.lefthand_publisher.Write(self.left_hand_cmd)
            self.righthand_publisher.Write(self.right_hand_cmd)

            time.sleep(self.config.control_dt)


    def execute_actions(self, action_chunk: Dict[str, Any]):
        left_arm_actions = action_chunk['action.left_arm']
        right_arm_actions = action_chunk['action.right_arm']
        left_hand_actions = action_chunk['action.left_hand']
        right_hand_actions = action_chunk['action.right_hand']

        # Stop any existing holding
        self.stop_holding_position()

        self._interpolate_to_first_action(action_chunk, interpolation_steps=10)


        for j in range(len(left_hand_actions)):
            # Apply velocity limiting to arm actions
            combined_arm_actions = np.concatenate([left_arm_actions[j], right_arm_actions[j]])

            # Split back into left and right
            left_arm_clipped = combined_arm_actions[:7]  # G1_29 has 7 joints per arm
            right_arm_clipped = combined_arm_actions[7:]

            self._set_arm_commands(left_arm_clipped, right_arm_clipped)
            self._set_hand_commands(left_hand_actions[j], right_hand_actions[j])

            # Send commands
            self.send_cmd(self.low_cmd)
            if not self.holding_position:
                print("LEFT_HAND_ACTION\n", left_hand_actions[j])
                print("RIGHT_HAND_ACTION\n", right_hand_actions[j])
            self.lefthand_publisher.Write(self.left_hand_cmd)
            self.righthand_publisher.Write(self.right_hand_cmd)

            # Sleep unless last action execution
            if j != (len(left_hand_actions) - 1):
                time.sleep(self.config.control_dt * 3)

        # Start holding the last executed positions
        last_left_arm = combined_arm_actions[:7]  # From last iteration
        last_right_arm = combined_arm_actions[7:]
        last_left_hand = left_hand_actions[-1]
        last_right_hand = right_hand_actions[-1]

        self.start_holding(last_left_arm, last_right_arm, last_left_hand, last_right_hand)

    def run(self):
        # Read robot state
        print("reading robots state")
        left_arm_state, right_arm_state = self._read_arm_state()
        left_hand_state, right_hand_state = self._read_hand_state()


        # Read camera
        camera_image = None
        if self.simulation:
            frame_rgb = self.sim_camera.get_latest_frame()
            if frame_rgb is not None:
                camera_image = np.expand_dims(frame_rgb, axis=0)
        else:
            camera_image = self.camera.get_frame()

        if camera_image is None or camera_image.size == 0:
            raise Exception("Could not get camera image!")

        # Build observation
        print("Building observation")
        print("LEFT_HAND_OBS", left_hand_state)
        print("RIGHT_HAND_OBS", right_hand_state)

        observation = self._build_observation(
            camera_image, left_arm_state, right_arm_state,
            left_hand_state, right_hand_state
        )

        print("Getting action")
        # Get policy prediction - during this time, the holding thread will
        # automatically keep sending the last known positions
        action_chunk = self.policy.get_action(observation)

        # Execute actions (this will stop the current holding and start new holding at the end)
        print("Executing action")
        self.execute_actions(action_chunk)
        print("Action executed!")

        time.sleep(0.06)

        # No sleep here - the holding thread handles position maintenance
        # Go immediately to next iteration for next policy call


if __name__ == "__main__":
    import argparse
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()

    sim = args.sim

    # Initialize DDS communication
    config_path = "deployment/configs/g1.yaml"
    if sim:
        print("configuring for sim")
        ChannelFactoryInitialize(1)
    else:
        ChannelFactoryInitialize(0)

    config = RobotConfig(config_path)
    controller = Controller(config, sim)

    try:
        # Enter the zero torque state, press the start key to continue executing
        if not sim:
            controller.zero_torque_state() # Move hands to the default position
            controller.move_to_default_pos()
            controller.move_hands_to_default()
            # Enter the default position state, press the A key to continue executing
            controller.default_pos_state()

        while True:
            try:
                print("running")
                controller.run()

                # Press the select key to exit
                if not sim:
                    if controller.remote_controller.button[KeyMap.select] == 1:
                        break
            except Exception as e:
                print(f"Error occurred: {e}")
                break
            except KeyboardInterrupt:
                break

    finally:
        # Clean up threads before exit
        controller.cleanup_threads()

        # Enter the damping state
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)
        print("Exit")
