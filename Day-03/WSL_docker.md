1. Docker 설치 & 시작
```sudo service docker start```
2. 계정에 Docker 권한 설정
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
3. 확인
```docker info```

4. 이미지 다운로드
```
docker pull neo4j:5.24.2-community
```

5. 도커 실행 명령어
```bash
cd Day-03/
sh 3_neo4j_docker_run.sh
```

6. 브라우저 접속
http://localhost:7474/browser/