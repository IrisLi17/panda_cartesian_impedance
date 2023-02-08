## Usage

Run

```shell
cd scripts/shell
# go to initial position
zsh two_arm_back.sh
# start the robot
zsh right_arm.sh
zsh left_arm.sh
```

Calibration

```shell
# change calibration file
cd projects/fairo/ploy...
conda activate poly...
python poli.../pytho/scri../launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.readonly=true
python poli.../pytho/scri../launch_camera.py width=1280 height=720 framerate=5 downsample=1 use_depth=false
python examples/calib.../calibration_service.py 

c # capture
...
c # capture
r # compute
s # save to calib_handeye.pkl data['base_T_cam']

python
from scipy.spatial.transform import Rotation as R
r = R.from_matrix(mat)
q = r.as_quat()

# change ROS config
cd ~/.ros/easy_handeye
vim panda_eob_calib_eye_on_base.yaml

```

```
roslaunch easy_handeye panda... robot_ip:=...
```



Reset

```shell
roslaunch franka_example_controllers move_to_start.launch robot_ip:=192.168.1.110

# recover from failure
rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"

# reset gripper
rostopic pub --once /franka_gripper/move/goal franka_gripper/MoveActionGoal "goal: { width: 0.08, speed: 0.1 }"
```

Pick and place

```
roslaunch panda_cartesian_impedance expert_pick_and_place.launch robot_ip:=192.168.1.110 markerSize:=0.05
```

