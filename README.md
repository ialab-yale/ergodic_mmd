# Ergodic MMD - Ergodic Exploration in Any Space

Ergodic MMD is an approach to generate ergodic trajectories in arbitrary domains only given sampled points from the search space. 

This code provides toy examples (as well as infrastructure) for [this paper](https://arxiv.org/abs/2410.10599).

## 3D Exploration

![showcase_3d](/imgs/Exploration_3D.png)

The `emmd_main.py` file allows you to construct ergodic trajectories over pointclouds and meshes. Run this code as-is to visualize trajectories over the surface of a tortoise. There are four other sample meshes included within the `/obj_files` directory, and if you would like to explore over your own 3D models, you may add them into this directory and modify the `mdl_path` parameter within the initialization of `emmd_main.py`. The basic requirement of this exploration is that there are points (or samples) available to construct a search manifold. As long as your model has points, it can be explored!

## 2D Exploration

![showcase_2d](/imgs/Exploration_2D.png)

Ergodic MMD also enables exploration in 2D environments. Running `ergodic_2d.py` as-is will plot a sample exploration in a 2D space with a non-uniform utility in the shape of a Y. We recommend that you modify this information distribution to your liking and see how it affects the trajectories! In this case, the points (or samples) that make the search manifold are randomly sampled from the 2D environment. 
