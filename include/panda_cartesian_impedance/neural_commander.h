#include <eigen_conversions/eigen_msg.h>
#include <torch/script.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <franka_msgs/FrankaState.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/Float32MultiArray.h>
#include <actionlib/client/simple_action_client.h>
#include <franka_gripper/GraspAction.h>
#include <franka_gripper/MoveAction.h>
#include <franka_gripper/HomingAction.h>


class NeuralCommander
{
private:
    void franka_state_callback(const franka_msgs::FrankaState::ConstPtr&);
    void marker_tf_callback();
    void timer_callback(const ros::TimerEvent&);
    void obs_callback(const std_msgs::Float32MultiArray::ConstPtr&);
    ros::Subscriber obs_sub;
    bool obs_received;
    std::vector<float> _obs;
    actionlib::SimpleActionClient<franka_gripper::GraspAction> grasp_client;
    actionlib::SimpleActionClient<franka_gripper::MoveAction> move_client;
    /* data */
    ros::NodeHandle node_handle;
    ros::Subscriber franka_state_sub;
    ros::Publisher cartesian_target_pub;
    tf::TransformListener tf_listener;
    ros::Timer timer;
    std::string marker_link_name;
    std::string ref_link_name;
    tf::Pose eef_pose;
    tf::Pose marker_pose;
    torch::jit::Module policy;
    torch::Tensor observation;
    torch::Tensor recurrent_hidden_state;
    torch::Tensor recurrent_mask;
    std::array<float, 4> action;
    geometry_msgs::PoseStamped cartesian_target_pose;
public:
    NeuralCommander(ros::NodeHandle* nodehandle);
    ~NeuralCommander();
    int step_counter;
    bool is_recurrent;
    int hidden_state_size;
    bool load_model(const char*);
    bool start();
};

NeuralCommander::NeuralCommander(ros::NodeHandle* nodehandle): 
    node_handle(*nodehandle), obs_received(false),
    grasp_client(*nodehandle, "franka_gripper/grasp", true),
    move_client(*nodehandle, "franka_gripper/move", true)
{
}

NeuralCommander::~NeuralCommander()
{
}
