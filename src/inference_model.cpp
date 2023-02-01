#include <ros/ros.h>
#include <eigen_conversions/eigen_msg.h>
#include <panda_cartesian_impedance/inference_model.h>
#include <tf_conversions/tf_eigen.h>


bool ModelInference::start() {
    // if (!node_handle.getParam("link_name", ref_link_name)) {
    //     ROS_ERROR_STREAM("param link_name not provided");
    //     return false;
    // }
    // node_handle.getParam("marker_link_name", marker_link_name);
    obs_sub = node_handle.subscribe(
        "rl_observation", 10, &ModelInference::obs_callback, this
    );
    //     "franka_gripper/joint_states", 10, &NeuralCommander::gripper_joint_callback, this
    // );
    action_pub = node_handle.advertise<std_msgs::Float32MultiArray>("rl_action", 10);
    return true;
}

void ModelInference::obs_callback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    _obs.clear();
    for (auto it=msg->data.begin(); it!=msg->data.end(); it++) {
        _obs.push_back(*it);
    }
    observation = torch::from_blob(_obs.data(), {(int)_obs.size()}, torch::kFloat32);
    observation = torch::unsqueeze(observation, 0);
    if (!obs_received){
        obs_received = true;
        std::cout << "observation received" << std::endl;
    } 

    std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(observation);
    inputs.push_back(torch::clone(observation));
    // inputs.push_back(torch::zeros_like(observation));
    
    // TODO: method name, control logic (flange) changed in isaac
    at::Tensor output = policy.get_method("take_action")(inputs).toTensor();
    auto output_a = output.accessor<float, 2>();
    for (int i=0; i<4; i++) {
        action[i] = output_a[0][i];
    }
    for (int i=0; i<4; i++) {
        if (action[i] > 1) action[i] = 1;
        else if (action[i] < -1) action[i] = -1;
    }
    // hack
    // if (step_counter % 5 == 0) action[3] = 1.0;
    // else action[3] = -1.0;

    std::cout << "cpp observation:" << inputs[0].toTensor() << std::endl;
    
    std::cout << "action: " << action[0] << ", " << action[1] << "," << action[2] << "," << action[3] << std::endl;
    
    action_msg.data.clear();
    for (int i=0; i<4; i++) {
        action_msg.data.push_back(action[i]);
    }
    action_pub.publish(action_msg);
}

bool ModelInference::load_model(const char* file_name) {
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
    ros::init(argc, argv, "model_inference");
    ros::NodeHandle node_handle;
    ModelInference predictor(&node_handle);
    if (!predictor.load_model(argv[1])) {
        return 1;
    }
    bool result = predictor.start();
    if (!result) {
        return 1;
    }
    ros::spin();
    return 0;
}
