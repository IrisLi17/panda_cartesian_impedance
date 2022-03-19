#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <franka_msgs/FrankaState.h>
#include <eigen_conversions/eigen_msg.h>
#include <panda_cartesian_impedance/neural_commander.h>
#include <tf_conversions/tf_eigen.h>


bool NeuralCommander::start() {
    // if (!node_handle.getParam("link_name", ref_link_name)) {
    //     ROS_ERROR_STREAM("param link_name not provided");
    //     return false;
    // }
    // node_handle.getParam("marker_link_name", marker_link_name);
    obs_sub = node_handle.subscribe(
        "rl_observation", 10, &NeuralCommander::obs_callback, this
    );
    franka_state_sub = node_handle.subscribe(
        "franka_state_controller/franka_states", 10, &NeuralCommander::franka_state_callback, this
    );
    cartesian_target_pub = node_handle.advertise<geometry_msgs::PoseStamped>("equilibrium_pose", 10);
    grasp_client.waitForServer();
    move_client.waitForServer();
    step_counter = 0;
    timer = node_handle.createTimer(ros::Duration(0.1), &NeuralCommander::timer_callback, this);
    return true;
}

void NeuralCommander::obs_callback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    _obs.clear();
    for (auto it=msg->data.begin(); it!=msg->data.end(); it++) {
        _obs.push_back(*it);
    }
    observation = torch::from_blob(_obs.data(), {(int)_obs.size()}, torch::kFloat32);
    observation = torch::unsqueeze(observation, 0);
    if (!obs_received){
        obs_received = true;
    } 
}

void NeuralCommander::franka_state_callback(const franka_msgs::FrankaState::ConstPtr& msg) {
    Eigen::Affine3d transform(Eigen::Matrix4d::Map(msg->O_T_EE.data()));
    tf::poseEigenToTF(transform, eef_pose);
}

// void NeuralCommander::marker_tf_callback() {
//     tf::StampedTransform transform;
//     tf_listener.waitForTransform(marker_link_name, ref_link_name, ros::Time(0), ros::Duration(10.0));
//     tf_listener.lookupTransform(marker_link_name, ref_link_name, ros::Time(0), transform);
//     marker_pose = tf::Pose(transform);
// }

void NeuralCommander::timer_callback(const ros::TimerEvent &e) {
    if (!obs_received) return;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(observation);
    at::Tensor output = policy.get_method("take_action"](inputs).toTensor();
    auto output_a = output.accessor<float, 2>();
    cartesian_target_pose.header.frame_id = ref_link_name;
    cartesian_target_pose.pose.position.x = eef_pose.getOrigin().getX() + output_a[0][0] * 0.05;
    cartesian_target_pose.pose.position.y = eef_pose.getOrigin().getY() + output_a[0][1] * 0.05;
    cartesian_target_pose.pose.position.z = eef_pose.getOrigin().getZ() + output_a[0][2] * 0.05;
    cartesian_target_pose.pose.orientation.x = 1.0;
    cartesian_target_pose.pose.orientation.y = 0.0;
    cartesian_target_pose.pose.orientation.z = 0.0;
    cartesian_target_pose.pose.orientation.w = 0.0;
    cartesian_target_pub.publish(cartesian_target_pose);
    // TODO: finger control
    float width = (output_a[0][3] + 1) * 0.04;
    if (output_a[0][3] < 0) {
        franka_gripper::GraspGoal goal;
        goal.width = width;
        goal.epsilon.inner = width;
        goal.epsilon.outer = 0.01;
        goal.force = 5;
        goal.speed = 0.1;
        grasp_client.sendGoal(goal);
    } else {
        franka_gripper::MoveGoal goal;
        goal.speed = 0.1;
        goal.width = width;
        move_client.sendGoal(goal);
    }
    step_counter += 1;
    if (step_counter >= 50) {
        timer.stop();
    }
}

bool NeuralCommander::load_model(const char* file_name) {
    try {
        policy = torch::jit::load(file_name);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return false;
    }
    std::cout << "load model ... ok\n";
    return true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "neural_commander");
    ros::NodeHandle node_handle;
    NeuralCommander neural_commander(&node_handle);
    if (!neural_commander.load_model(argv[1])) {
        return 1;
    }
    bool result = neural_commander.start();
    if (!result) {
        return 1;
    }
    ros::spin();
    return 0;
}
