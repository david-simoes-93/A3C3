# Asynchronous Advantage Actor-Centralized-Critic with Communication (A3C3)

A distributed asynchronous actor-critic algorithm in a multi-agent setting with differentiable communication and a centralized
critic.

Check out learned policies here: https://youtu.be/fB71yKcP3iU

![Untitled](https://user-images.githubusercontent.com/9117323/73293622-2108a480-41fc-11ea-97b1-5b172ac5b857.png)

Contains 4 environment suites:

- POC Suite: Hidden Reward, Navigation, Pursuit, Traffic Intersection
- MPE Suite: Cooperative Navigation, Cooperative Communication, Cooperative Reference, Tag
- KiloBot Suite: Light, Join, Split
- 3d Soccer Simulation Suite: Passing, Keep-Away

Also contains scripts to launch A3C3 and learn policies. Use the `requirements.txt` to install your dependencies and run the scripts.

Each agent is defined by 3 networks.

![Untitled2](https://user-images.githubusercontent.com/9117323/73293624-2108a480-41fc-11ea-96fe-12ea3bcb2e33.png)

The algorithm is distributed, and multiple workers update the networks.

![Untitled3](https://user-images.githubusercontent.com/9117323/73293625-2108a480-41fc-11ea-8bc9-e16bfa6086da.png)

The actor network learns a local policy.

![Untitled4](https://user-images.githubusercontent.com/9117323/73293626-21a13b00-41fc-11ea-96d7-b6b9da092804.png)

The centralized critic evaluates the policy.

![Untitled5](https://user-images.githubusercontent.com/9117323/73293627-21a13b00-41fc-11ea-96b9-40089c781fde.png)

The communicator network learns a communication protocol between agents.

![Untitled6](https://user-images.githubusercontent.com/9117323/73293628-21a13b00-41fc-11ea-93b9-72c3756f61bd.png)
