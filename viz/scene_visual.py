import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad

from PIL import Image, ImageDraw
from gym_IFR.envs.relation import *


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def position_to_tuple(position):
    return (position["x"], position["y"], position["z"])


# def draw_self_loop(ax, center, radius, facecolor='#2693de', edgecolor='#000000', theta1=-30, theta2=180):
def draw_self_loop(ax, center, rwidth, radius=7., facecolor='#2693de', edgecolor='#000000', theta1=-10, theta2=280):
    # Add the ring
    ring = mpatches.Wedge(center, radius, theta1, theta2, width=rwidth)
    # Triangle edges
    # offset = 2
    offset = 5
    xcent = center[0] - radius + (rwidth/2)
    left = [xcent - offset, center[1]]
    right = [xcent + offset, center[1]]
    bottom = [(left[0]+right[0])/2., center[1]-0.05]
    arrow = plt.Polygon([left, right, bottom, left])
    p = PatchCollection(
        [ring, arrow],
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    ax.add_collection(p)


def render_relation_in_graph(frame_meta, frame, action_list, obj_names, obj_pos, gt, pred, known, output_path):
    DG = nx.MultiDiGraph()
    node_colors = []
    for i, name in enumerate(obj_names):
        name = name.split(':')[0]
        obj_names[i] = name
        if not (gt[i, :] == 1).any() and not (gt[:, i] == 1).any() and not (pred[i, :] == 1).any() and not (pred[:, i] == 1).any() and not (known[i, :] == 1).any() and not (known[:, i] == 1).any() and i not in action_list:
            continue

        if i in action_list:
            DG.add_node(name)
            node_colors.append('tab:green')
        else:
            DG.add_node(name)
            node_colors.append('tab:blue')

    sources = set()
    colors = []
    weights = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] == 1:
                DG.add_edge(obj_names[i], obj_names[j])
                sources.add(obj_names[i])
                colors.append('tab:blue')
                weights.append(8)

            if known[i, j] == 1:
                DG.add_edge(obj_names[i], obj_names[j])
                sources.add(obj_names[i])
                colors.append('tab:green')
                weights.append(8)

            if pred[i, j] == 1:
                DG.add_edge(obj_names[i], obj_names[j])
                sources.add(obj_names[i])
                colors.append('tab:red')
                weights.append(2)

    # plt.figure(figsize=(10, 3))
    plt.figure(figsize=(10, 10))
    # plt.figure(figsize=(8,18))
    # pos = nx.bipartite_layout(DG, nodes=list(sources), aspect_ratio=1.)
    pos = nx.bipartite_layout(DG, nodes=list(sources), align='horizontal')
    # pos = nx.circular_layout(DG)
    # edges = DG.edges()
    # colors = [DG[u][v]['color'] for u,v in edges]
    # weights = [DG[u][v]['weight'] for u,v in edges]
    # colors = [DG[u][v][0]['color'] for u,v in edges]
    # weights = [DG[u][v][0]['weight'] for u,v in edges]

    # DG.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
    # DG.graph['graph'] = {'scale': '3'}

    nx.draw(DG, pos, node_color=node_colors, edge_color=colors, width=weights, with_labels=False, node_size=1000, linewidths=1, alpha=0.8, arrowsize=10)
    text = nx.draw_networkx_labels(DG, pos, font_size=22, font_color='whitesmoke', font_weight='bold', alpha=0.8)
    for _, t in text.items():
        t.set_rotation('vertical')

    plt.savefig(output_path)
    plt.close()

def render_relation_in_scene(frame_meta, frame, action_list, obj_pos, gt, pred, known, scene_idx_dict, output_path):
    cam_position = frame_meta["cam_position"]
    cam_orth_size = frame_meta["cam_orth_size"]
    frame_shape = frame_meta["frame_shape"]
    pos_translator = ThorPositionTo2DFrameTranslator(frame_shape, position_to_tuple(cam_position), cam_orth_size)

    obj_pixel_pos = []
    for pos in obj_pos:
        pos = (pos[0], pos[2])
        obj_pixel_pos.append(tuple(reversed(pos_translator(pos))))

    # fig = plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.imshow(frame)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # if action is not None:
    #     plt.plot(obj_pixel_pos[action][0], obj_pixel_pos[action][1], marker='o', markersize=10)


    # ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    trans = 0.8
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] == 1 and i != j:
                plt.annotate("", obj_pixel_pos[i], obj_pixel_pos[j], arrowprops=dict(arrowstyle="<-", color=(0, 0, 1, trans), lw=7))
            if gt[i, j] == 1 and i == j:
                draw_self_loop(ax, center=obj_pixel_pos[i], rwidth=4., facecolor=(0, 0, 1, trans), edgecolor=None)

            if known[i, j] == 1 and i != j:
                plt.annotate("", obj_pixel_pos[i], obj_pixel_pos[j], arrowprops=dict(arrowstyle="<-", color=(0, 1, 0, trans), lw=5))
            if known[i, j] == 1 and i == j:
                draw_self_loop(ax, center=obj_pixel_pos[i], rwidth=3., facecolor=(0, 1, 0, trans), edgecolor=None)

            if pred[i, j] == 1 and i != j:
                plt.annotate("", obj_pixel_pos[i], obj_pixel_pos[j], arrowprops=dict(arrowstyle="<-", color=(1, 0, 0, trans), lw=3))
            if pred[i, j] == 1 and i == j:
                draw_self_loop(ax, center=obj_pixel_pos[i], rwidth=2, facecolor=(1, 0, 0, trans), edgecolor=None)

    if action_list is not None:
        for action in action_list:
            plt.plot(obj_pixel_pos[action][0], obj_pixel_pos[action][1], marker='X', color=(0, 1, 0, 0.6), markersize=16)

    texts = []
    for i, pos in enumerate(obj_pixel_pos):
        # if not (gt[i, :] == 1).any() and not (gt[:, i] == 1).any() and not (pred[i, :] == 1).any() and not (pred[:, i] == 1).any() and not (known[i, :] == 1).any() and not (known[:, i] == 1).any() and i not in action_list:
        #     continue
        # else:
        #     if i in scene_idx_dict:
        #         idx = scene_idx_dict[i]
        #     else:
        #         idx = len(scene_idx_dict)
        #         scene_idx_dict[i] = idx
        texts.append(plt.text(pos[0], pos[1], str(i), color=(1, 1, 0), fontsize=20))
        # texts.append(plt.text(pos[0], pos[1], str(idx), color=(1, 1, 0), fontsize=20))
    adjust_text(texts)

    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    path = '../data/ai2thor_scenes/Kitchen/train/FloorPlan10_1'
    with open(f'{path}/frame_meta.json', 'r') as f:
        frame_meta = json.load(f)

    # img = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img = Image.open(f"{path}/frame.png")

    obj_name = np.load(f'{path}/name.npy')
    obj_pos = np.load(f'{path}/pos.npy')
    plt.imshow(img)
    plt.show()
