# -*- coding: utf-8 -*-

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt

# generate mesh
def mesh_to_img(inst_mesh):
    mesh = pyrender.Mesh.from_trimesh(inst_mesh, smooth=False)

    # compose scene
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[1, 1, 1])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    c = 2**-0.5
    scene.add(camera, pose=[[ 1,  0,  0,  0],
                            [ 0,  c, -c, -0.2],
                            [ 0,  c,  c,  0.2],
                            [ 0,  0,  0,  1]])

    # render scene
    r = pyrender.OffscreenRenderer(512, 512)
    img, _ = r.render(scene)
    return img

