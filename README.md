# MineRL
![Build Passing](https://github.com/will-maclean/MineRL/workflows/main%20workflow/badge.svg)
![Testing Coverage](./coverage.svg)

Run example with CartPole:

```
python src/scripts/train.py --policy cartpole-dqn --env CartPole-v0 --render
```

Note that you will generally need to install `pyglet` to render CartPole. This can be done easily with `pip install pyglet`.

## How to run with docker:
<br />

### simply:
```
docker run -d amirtoosi/minerl
```
<br />

## Other useful docker commands:
<br />

### pull the image from dockerhub
```
docker pull amirtoosi/minerl
```
### Build the image locally
```
docker build . -t <image_name>
```
### Run the image in a new container
```
docker run -d <image_name>
```
### Look at the list of running containers
```
docker ps
```
### See the logs for the running container
```
docker logs --follow <container_name>
```
### Get an interactive shell inside the container
```
docker exec -it <container_name> sh
```
