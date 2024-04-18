from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch RL env controller
    launch_rl_env = Node(
        package="dqn_agent",
        executable="main.py",
        name="dqn_agent",
        output="screen",
    )

    ld = LaunchDescription()
    ld.add_action(launch_rl_env)

    return ld
