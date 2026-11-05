#include "rebotarm_ros2_control/uart_system.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace rebotarm_ros2_control
{
using CallbackReturn = hardware_interface::CallbackReturn;
using Return = hardware_interface::return_type;

UartSystem::~UartSystem() { stop_worker(); }

CallbackReturn UartSystem::on_init(const hardware_interface::HardwareInfo & info)
{
  if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {return CallbackReturn::ERROR;}
  try {
    if (info.joints.size() != 6) {throw std::invalid_argument("exactly six arm joints required");}
    namespace_ = info.hardware_parameters.at("arm_namespace");
    while (!namespace_.empty() && namespace_.front() == '/') {namespace_.erase(0, 1);}
    while (!namespace_.empty() && namespace_.back() == '/') {namespace_.pop_back();}
    if (namespace_.empty()) {throw std::invalid_argument("empty arm_namespace");}
    auto parameter = [&](const std::string & name, double fallback) {
      const auto entry = info.hardware_parameters.find(name);
      return entry == info.hardware_parameters.end() ? fallback : std::stod(entry->second);
    };
    publish_hz_ = parameter("command_hz", 50.);
    feedback_timeout_ = parameter("feedback_timeout_sec", .25);
    command_timeout_ = parameter("command_timeout_sec", .2);
    if (!std::isfinite(publish_hz_) || publish_hz_ < 1. || publish_hz_ > 500. ||
      !std::isfinite(feedback_timeout_) || feedback_timeout_ <= 0. ||
      !std::isfinite(command_timeout_) || command_timeout_ <= 0.)
    {
      throw std::invalid_argument("invalid rates or timeouts");
    }
    for (size_t i = 0; i < 6; ++i) {
      const auto & joint = info.joints[i];
      joint_names_[i] = "joint" + std::to_string(i + 1);
      if (joint.name != joint_names_[i] || joint.command_interfaces.size() != 1 ||
        joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION)
      {
        throw std::invalid_argument("expected joint1..joint6, each with position command interface");
      }
      for (const auto & name : {"position", "velocity", "effort"}) {
        if (std::none_of(joint.state_interfaces.begin(), joint.state_interfaces.end(),
          [&](const auto & interface) {return interface.name == name;}))
        {
          throw std::invalid_argument(joint.name + " must export position/velocity/effort state");
        }
      }
      lower_[i] = std::stod(joint.command_interfaces[0].min);
      upper_[i] = std::stod(joint.command_interfaces[0].max);
      speed_[i] = std::stod(joint.parameters.at("velocity_limit"));
      if (!std::isfinite(lower_[i]) || !std::isfinite(upper_[i]) || lower_[i] >= upper_[i] ||
        !std::isfinite(speed_[i]) || speed_[i] <= 0.)
      {
        throw std::invalid_argument(joint.name + " has invalid bounds or velocity_limit");
      }
    }
  } catch (const std::exception & exc) {
    RCLCPP_ERROR(rclcpp::get_logger("rebotarm_uart_system"), "%s", exc.what());
    return CallbackReturn::ERROR;
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  position_.fill(nan); velocity_.fill(nan); effort_.fill(nan); command_.fill(nan);
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> UartSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> result;
  for (size_t i = 0; i < 6; ++i) {
    result.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &position_[i]);
    result.emplace_back(joint_names_[i], hardware_interface::HW_IF_VELOCITY, &velocity_[i]);
    result.emplace_back(joint_names_[i], hardware_interface::HW_IF_EFFORT, &effort_[i]);
  }
  return result;
}

std::vector<hardware_interface::CommandInterface> UartSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> result;
  for (size_t i = 0; i < 6; ++i) {
    result.emplace_back(joint_names_[i], hardware_interface::HW_IF_POSITION, &command_[i]);
  }
  return result;
}

bool UartSystem::fresh(const State & state, Clock::time_point now) const
{
  return state.complete && std::chrono::duration<double>(now - state.measured_at).count() <= feedback_timeout_ &&
         std::chrono::duration<double>(now - state.status_at).count() <= feedback_timeout_;
}

CallbackReturn UartSystem::on_configure(const rclcpp_lifecycle::State &)
{
  start_worker();
  RCLCPP_INFO(node_->get_logger(), "Configured inactive; activation requires fresh six-axis feedback and explicit UART enable");
  return CallbackReturn::SUCCESS;
}

CallbackReturn UartSystem::on_activate(const rclcpp_lifecycle::State &)
{
  State state;
  {std::lock_guard<std::mutex> lock(state_mutex_); state = received_;}
  if (!fresh(state, Clock::now()) || !state.enabled) {
    RCLCPP_ERROR(node_->get_logger(), "Activation refused: require fresh joint_states and enabled ArmStatus; no zero state is fabricated");
    return CallbackReturn::FAILURE;
  }
  position_ = state.position; velocity_ = state.velocity; effort_ = state.effort;
  command_ = state.position;
  {std::lock_guard<std::mutex> lock(command_mutex_); outgoing_.valid = false;}
  reactivation_required_.store(false);
  active_.store(true);
  RCLCPP_INFO(node_->get_logger(), "Activated at measured position; this plugin never calls enable/disable");
  return CallbackReturn::SUCCESS;
}

CallbackReturn UartSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  active_.store(false); reactivation_required_.store(true);
  return CallbackReturn::SUCCESS;
}
CallbackReturn UartSystem::on_cleanup(const rclcpp_lifecycle::State &) {stop_worker(); return CallbackReturn::SUCCESS;}
CallbackReturn UartSystem::on_shutdown(const rclcpp_lifecycle::State &) {stop_worker(); return CallbackReturn::SUCCESS;}

Return UartSystem::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  // try_lock only: no ROS callback, serial I/O, publication or waiting in update().
  std::unique_lock<std::mutex> lock(state_mutex_, std::try_to_lock);
  if (lock.owns_lock() && fresh(received_, Clock::now())) {
    position_ = received_.position; velocity_ = received_.velocity; effort_ = received_.effort;
  }
  return Return::OK;
}

Return UartSystem::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (!active_.load() || reactivation_required_.load()) {return Return::OK;}
  std::unique_lock<std::mutex> lock(command_mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {return Return::OK;}
  for (size_t i = 0; i < 6; ++i) {
    if (!std::isfinite(command_[i]) || command_[i] < lower_[i] || command_[i] > upper_[i]) {
      outgoing_.valid = false;
      return Return::OK;
    }
  }
  outgoing_.position = command_; outgoing_.at = Clock::now(); outgoing_.valid = true;
  return Return::OK;
}

void UartSystem::receive_state(const sensor_msgs::msg::JointState & message)
{
  State next;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  next.effort.fill(nan);
  for (size_t i = 0; i < 6; ++i) {
    const auto found = std::find(message.name.begin(), message.name.end(), joint_names_[i]);
    if (found == message.name.end()) {return;}
    const auto index = std::distance(message.name.begin(), found);
    if (index >= static_cast<long>(message.position.size()) || index >= static_cast<long>(message.velocity.size()) ||
      !std::isfinite(message.position[index]) || !std::isfinite(message.velocity[index])) {return;}
    next.position[i] = message.position[index]; next.velocity[i] = message.velocity[index];
    if (index < static_cast<long>(message.effort.size())) {next.effort[i] = message.effort[index];}
  }
  std::lock_guard<std::mutex> lock(state_mutex_);
  received_.position = next.position; received_.velocity = next.velocity; received_.effort = next.effort;
  received_.measured_at = Clock::now(); received_.complete = true;
}

void UartSystem::receive_status(const rebotarm_msgs::msg::ArmStatus & message)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  received_.enabled = message.enabled; received_.status_at = Clock::now();
}

void UartSystem::start_worker()
{
  stop_worker();
  {std::lock_guard<std::mutex> lock(state_mutex_); received_ = State{};}
  {std::lock_guard<std::mutex> lock(command_mutex_); outgoing_ = Command{};}
  rclcpp::NodeOptions options;
  options.use_global_arguments(false);  // CM's __node remap must not rename this private node.
  node_ = std::make_shared<rclcpp::Node>("rebotarm_uart_system_io", options);
  const std::string prefix = "/" + namespace_;
  state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    prefix + "/joint_states", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr message) {receive_state(*message);});
  status_sub_ = node_->create_subscription<rebotarm_msgs::msg::ArmStatus>(
    prefix + "/arm_status", rclcpp::QoS(1).reliable().transient_local(),
    [this](rebotarm_msgs::msg::ArmStatus::ConstSharedPtr message) {receive_status(*message);});
  // Keep one batch command contract for every producer.  The planning task
  // server and posvel_to_mit both use /arm/command/position; publishing to a
  // shorter legacy topic here would silently leave the UART converter idle.
  command_pub_ = node_->create_publisher<rebotarm_msgs::msg::ArmPositionCommand>(
    prefix + "/arm/command/position", rclcpp::QoS(1));
  executor_.add_node(node_);
  stop_.store(false);
  worker_ = std::thread([this]() {run();});
}

void UartSystem::stop_worker()
{
  active_.store(false); reactivation_required_.store(true); stop_.store(true);
  if (worker_.joinable()) {worker_.join();}
  if (node_) {executor_.remove_node(node_);}
  command_pub_.reset(); state_sub_.reset(); status_sub_.reset(); node_.reset();
}

void UartSystem::run()
{
  auto next_publish = Clock::now();
  while (!stop_.load() && rclcpp::ok()) {
    executor_.spin_some();
    State state;
    Command command;
    {std::lock_guard<std::mutex> lock(state_mutex_); state = received_;}
    {std::lock_guard<std::mutex> lock(command_mutex_); command = outgoing_;}
    const auto now = Clock::now();
    if (active_.load() && (!fresh(state, now) || !state.enabled)) {
      if (!reactivation_required_.exchange(true)) {
        RCLCPP_ERROR(node_->get_logger(), "Command stream stopped: feedback/status stale or disabled. Deactivate controller and hardware, then explicitly reactivate at measured position");
      }
    }
    if (active_.load() && !reactivation_required_.load() && command.valid && fresh(state, now) && state.enabled &&
      std::chrono::duration<double>(now - command.at).count() <= command_timeout_ && now >= next_publish)
    {
      rebotarm_msgs::msg::ArmPositionCommand message;
      message.stamp = node_->now(); message.positions = command.position; message.velocity_limits = speed_;
      command_pub_->publish(message);
      next_publish = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1. / publish_hz_));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
}  // namespace rebotarm_ros2_control
PLUGINLIB_EXPORT_CLASS(rebotarm_ros2_control::UartSystem, hardware_interface::SystemInterface)
