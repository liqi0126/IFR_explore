import pptk
import numpy as np
import sapien.core as sapien
from gym_sapien.envs.relation import InDoorObject, get_model
from gym_sapien.envs.util import get_global_mesh


def main():
    engine = sapien.Engine(0, 0.001, 0.005)
    renderer = None
    controller = None
    # render_config = sapien.OptifuserConfig()
    # render_config.use_ao = True
    # renderer = sapien.OptifuserRenderer(config=render_config)
    # controller = sapien.OptifuserController(renderer)
    # engine.set_renderer(renderer)

    config = sapien.SceneConfig()
    config.gravity = [0, 0, 0]
    config.solver_iterations = 15
    config.solver_velocity_iterations = 2
    config.enable_pcm = False

    scene = engine.create_scene(config=config)
    scene.set_timestep(1 / 200)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = 0.5

    for OBJ in InDoorObject.__subclasses__():
        if OBJ.idx == 3:
            if OBJ.sapien_id is not None:
                print(OBJ.__name__)
                for idx in OBJ.sapien_id:
                    print(idx)
                    model = get_model(loader, idx)
                    pc = get_global_mesh(model, 2048)
                    v = pptk.viewer(pc)
                    v.set(point_size=0.005)
                    v.set(phi=6.89285231)
                    v.set(r=2.82503915)
                    v.set(theta=0.91104609)
                    v.set(lookat=[0.00296851, 0.01331535,-0.47486299])
                    import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    main()