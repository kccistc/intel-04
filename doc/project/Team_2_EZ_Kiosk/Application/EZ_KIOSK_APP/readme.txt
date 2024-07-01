Cmake에 
# Qt 경로 설정
set(Qt6_DIR "/home/ubuntu/Qt/6.7.2/gcc_64/lib/cmake/Qt6")

OpenGL 종속성 설치:
sudo apt-get update
sudo apt-get install libgl1-mesa-dev
sudo apt-get install libglu1-mesa-dev

gobject-introspection-1.0 종속성
sudo apt-get update
sudo apt-get install -y \
    gobject-introspection \
    libgirepository1.0-dev \
    build-essential \
    meson \
    pkg-config \
    python3-dev

