curl -L "http://192.168.3.248:8081/artifactory/HighWayRelease/pillarx_sdk/artifacts/docker/Miniconda3-latest-Linux-x86_64.sh" --output Miniconda3-latest-Linux-x86_64.sh
curl -L "http://192.168.3.248:8081/artifactory/HighWayRelease/pillarx_sdk/artifacts/docker/ cmake-3.25.1-linux-x86_64.sh" --output  cmake-3.25.1-linux-x86_64.sh
docker build -t 192.168.3.248:8081/docker/pillarx_builder:0.2 .