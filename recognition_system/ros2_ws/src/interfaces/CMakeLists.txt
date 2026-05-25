cmake_minimum_required(VERSION 3.8)
project(interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonDetected.msg"
  "srv/FindByFace.srv"
  "srv/CreateOrUpdateGuest.srv"
  "srv/AppendConversation.srv"
  "srv/RecordAudio.srv"
  "srv/Speak.srv"
  DEPENDENCIES std_msgs
)

ament_package()
