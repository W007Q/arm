#ifndef REBOTARM_ROS2_CONTROL__UART_SYSTEM_HPP_
#define REBOTARM_ROS2_CONTROL__UART_SYSTEM_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rebotarm_msgs/msg/arm_position_command.hpp"
#include "rebotarm_msgs/msg/arm_status.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace rebotarm_ros2_control
{
// Only the existing rebotarm_uart worker opens the serial device. This plugin
// bridges standard ros2_control interfaces to its ROS batch command interface.
class UartSystem : public hardware_interface::SystemInterface
{
public:
  ~UartSystem() override;
  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo &) override;
  hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override;
  hardware_interface::CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::return_type read(const rclcpp::Time &, const rclcpp::Duration &) override;
  hardware_interface::return_type write(const rclcpp::Time &, const rclcpp::Duration &) override;

private:
  using Clock = std::chrono::steady_clock;
  struct State
  {
    std::array<double, 6> position{}, velocity{}, effort{};
    Clock::time_point measured_at{}, status_at{};
    bool complete{false}, enabled{false};
  };
  struct Command
  {
    std::array<double, 6> position{};
    Clock::time_point at{};
    bool valid{false};
  };
  bool fresh(const State &, Clock::time_point) const;
  void start_worker();
  void stop_worker();
  void receive_state(const sensor_msgs::msg::JointState &);
  void receive_status(const rebotarm_msgs::msg::ArmStatus &);
  void run();
  std::array<double, 6> position_, velocity_, effort_, command_, lower_, upper_, speed_;
  std::array<std::string, 6> joint_names_;
  std::string namespace_;
  double publish_hz_{50.}, feedback_timeout_{.25}, command_timeout_{.2};
  std::mutex state_mutex_, command_mutex_;
  State received_;
  Command outgoing_;
  std::atomic<bool> stop_{false}, active_{false}, reactivation_required_{true};
  std::thread worker_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::Subscription<rebotarm_msgs::msg::ArmStatus>::SharedPtr status_sub_;
  rclcpp::Publisher<rebotarm_msgs::msg::ArmPositionCommand>::SharedPtr command_pub_;
};
}  // namespace rebotarm_ros2_control
#endif
