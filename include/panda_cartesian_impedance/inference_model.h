#include <eigen_conversions/eigen_msg.h>
#include <torch/script.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

class ModelInference
{
private:
    void obs_callback(const std_msgs::Float32MultiArray::ConstPtr&);
    // void gripper_joint_callback(const sensor_msgs::JointState::ConstPtr&);
    ros::Subscriber obs_sub;
    bool obs_received;
    std::vector<float> _obs;
    /* data */
    ros::NodeHandle node_handle;
    ros::Publisher action_pub;
    torch::jit::Module policy;
    torch::Tensor observation;
    std::array<float, 4> action;
public:
    ModelInference(ros::NodeHandle* nodehandle);
    ~ModelInference();
    int step_counter;
    bool load_model(const char*);
    bool start();
};

ModelInference::ModelInference(ros::NodeHandle* nodehandle): 
    node_handle(*nodehandle), obs_received(false)
{
}

ModelInference::~ModelInference()
{
}
