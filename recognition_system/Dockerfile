FROM ros:jazzy

RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libportaudio2 \
    libpulse0 \
    libasound2-plugins \
    ros-jazzy-ros2launch \
    ros-jazzy-launch \
    ros-jazzy-launch-ros \
    && rm -rf /var/lib/apt/lists/*

# Ursprung von asound.conf: https://www.freedesktop.org/wiki/Software/PulseAudio/Documentation/User/PerfectSetup/
# /etc weil ALSA die systemweite Konfiguration automatisch von dort lädt
COPY asound.conf /etc/asound.conf

WORKDIR /ros2_ws
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --ignore-installed -r /tmp/requirements.txt

COPY ros2_ws/src /ros2_ws/src

RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/jazzy/setup.sh && colcon build

CMD ["/bin/bash", "-c", \
    "source /opt/ros/jazzy/setup.bash && \
     source /ros2_ws/install/setup.bash && \
     ros2 launch system_launch launch.py"]
