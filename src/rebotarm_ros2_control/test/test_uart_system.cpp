// Synthetic feedback only: this test never creates a UART bridge or opens a bus.
#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "rebotarm_ros2_control/uart_system.hpp"

namespace
{
using namespace std::chrono_literals;
using CallbackReturn = hardware_interface::CallbackReturn;

hardware_interface::HardwareInfo hardware_info()
{
  hardware_interface::HardwareInfo info;
  info.name = "TestUart";
  info.type = "system";
  info.hardware_parameters = {{"arm_namespace", "uart_system_test"},
    {"command_hz", "50"}, {"feedback_timeout_sec", "0.1"},
    {"command_timeout_sec", "0.05"}};
  for (size_t i = 0; i < 6; ++i) {
    hardware_interface::ComponentInfo joint;
    joint.name = "joint" + std::to_string(i + 1);
    joint.parameters["velocity_limit"] = "0.2";
    hardware_interface::InterfaceInfo command;
    command.name = "position";
    command.min = "-1.0";
    command.max = "1.0";
    joint.command_interfaces.push_back(command);
    for (const auto & name : {"position", "velocity", "effort"}) {
      hardware_interface::InterfaceInfo state;
      state.name = name;
      joint.state_interfaces.push_back(state);
    }
    info.joints.push_back(joint);
  }
  return info;
}

class UartSystemTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    node_ = std::make_shared<rclcpp::Node>("synthetic_uart_feedback");
    states_ = node_->create_publisher<sensor_msgs::msg::JointState>(
      "/uart_system_test/joint_states", rclcpp::SensorDataQoS());
    status_ = node_->create_publisher<rebotarm_msgs::msg::ArmStatus>(
      "/uart_system_test/arm_status", rclcpp::QoS(1).reliable().transient_local());
    commands_ = node_->create_subscription<rebotarm_msgs::msg::ArmPositionCommand>(
      "/uart_system_test/arm/command/position", 10,
      [this](rebotarm_msgs::msg::ArmPositionCommand::ConstSharedPtr message) {
        last_command_ = *message; ++command_count_;
      });
    ASSERT_EQ(system_.on_init(hardware_info()), CallbackReturn::SUCCESS);
    ASSERT_EQ(system_.on_configure(rclcpp_lifecycle::State{}), CallbackReturn::SUCCESS);
  }

  void TearDown() override {system_.on_cleanup(rclcpp_lifecycle::State{});}

  void pump(std::chrono::milliseconds duration, bool feedback, bool write = false)
  {
    const auto deadline = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < deadline) {
      if (feedback) {
        sensor_msgs::msg::JointState state;
        for (size_t i = 0; i < 6; ++i) {
          state.name.push_back("joint" + std::to_string(i + 1));
          state.position.push_back(.05 * static_cast<double>(i + 1));
          state.velocity.push_back(0.);
          state.effort.push_back(0.);
        }
        states_->publish(state);
        rebotarm_msgs::msg::ArmStatus status;
        status.enabled = true;
        status_->publish(status);
      }
      if (write) {
        system_.read(rclcpp::Time(0), rclcpp::Duration::from_seconds(.01));
        system_.write(rclcpp::Time(0), rclcpp::Duration::from_seconds(.01));
      }
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(5ms);
    }
  }

  rebotarm_ros2_control::UartSystem system_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr states_;
  rclcpp::Publisher<rebotarm_msgs::msg::ArmStatus>::SharedPtr status_;
  rclcpp::Subscription<rebotarm_msgs::msg::ArmPositionCommand>::SharedPtr commands_;
  rebotarm_msgs::msg::ArmPositionCommand last_command_;
  size_t command_count_{0};
};

TEST_F(UartSystemTest, RequiresFeedbackAndStartsAtMeasuredPosition)
{
  EXPECT_EQ(system_.on_activate(rclcpp_lifecycle::State{}), CallbackReturn::FAILURE);
  pump(200ms, true);
  ASSERT_EQ(system_.on_activate(rclcpp_lifecycle::State{}), CallbackReturn::SUCCESS);
  pump(100ms, true, true);
  ASSERT_GT(command_count_, 0u);
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_DOUBLE_EQ(last_command_.positions[i], .05 * static_cast<double>(i + 1));
    EXPECT_DOUBLE_EQ(last_command_.velocity_limits[i], .2);
  }
}

TEST_F(UartSystemTest, PublishesBatchTargetsAndLatchesAfterFeedbackLoss)
{
  pump(200ms, true);
  ASSERT_EQ(system_.on_activate(rclcpp_lifecycle::State{}), CallbackReturn::SUCCESS);
  auto commands = system_.export_command_interfaces();
  for (size_t i = 0; i < commands.size(); ++i) {commands[i].set_value(.4);}
  pump(100ms, true, true);
  ASSERT_GT(command_count_, 0u);
  for (double position : last_command_.positions) {EXPECT_DOUBLE_EQ(position, .4);}

  pump(200ms, false, true);  // No new CAN-derived state must stop output.
  const auto stopped = command_count_;
  pump(100ms, true, true);   // Recovered input alone must not restart motion.
  EXPECT_EQ(command_count_, stopped);

  ASSERT_EQ(system_.on_deactivate(rclcpp_lifecycle::State{}), CallbackReturn::SUCCESS);
  ASSERT_EQ(system_.on_activate(rclcpp_lifecycle::State{}), CallbackReturn::SUCCESS);
  pump(100ms, true, true);
  EXPECT_GT(command_count_, stopped);
  // Explicit reactivation discards the stale .4-rad target.
  EXPECT_DOUBLE_EQ(last_command_.positions[0], .05);
}
}  // namespace
