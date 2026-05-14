# Docker 

构建镜像
```shell
sh docker.sh
```

运行容器
```shell
docker run --gpus all -itd --name test_pillarx 192.168.3.248:8081/docker/pillarx:0.2 bash
docker exec -it test_pillarx bash
```

上传镜像到JFROG
```shell
docker push 192.168.3.248:8081/docker/pillarx_builder:$TAG
```