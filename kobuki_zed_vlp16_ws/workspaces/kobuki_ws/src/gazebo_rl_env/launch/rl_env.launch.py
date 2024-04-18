from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ARGUMENTS = [
    DeclareLaunchArgument(
        "launch_rviz",
        default_value="False",
        description="Launch rviz2, by default is False",
    ),
]


def generate_launch_description():
    # Launch Gazebo
    launch_gazebo = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("kobuki_gazebo"), "launch", "tb3_world.launch.py"],
        )
    )

    # Launch Gazebo RL env controller
    launch_gazebo_rl_env = Node(
        package="gazebo_rl_env",
        executable="main.py",
        name="gazebo_rl_env",
        output="screen",
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(launch_gazebo)
    ld.add_action(launch_gazebo_rl_env)

    return ld
