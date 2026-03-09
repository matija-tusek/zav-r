import math
import random
import pygad
import numpy as np
import json
from typing import Dict, List, Any


def genome_from_genes(genes, NUM_LEGS):
    genome = {
        "creature_name": f"creature_{np.random.randint(1000,9999)}",
        "global_params": {
            "num_legs": NUM_LEGS,
            "symmetry": "bilateral"
        },
        "base_body": {
            "geometry": {"type": "box",
                         "size": {"x": genes[0], "y": genes[1], "z": genes[2]}},
            "inertial": {"mass": genes[3],
                         "inertia": {"ixx": genes[4], "iyy": genes[5], "izz": genes[6]}},
            "visual": {"color": {"rgba": [0.4, 0.6, 0.8, 1.0]}},  # fixed colour, not a gene
            "base_height": 1.5
        },
        "leg_params": []
    }

    # 7 body genes (indices 0-6), then 14 per leg
    leg_gene_start   = 7
    leg_genes_per_leg = 14

    for i in range(NUM_LEGS):
        idx = leg_gene_start + i * leg_genes_per_leg
        leg = {
            "leg_id": i,
            "leg_name": f"{'left' if i % 2 == 0 else 'right'}Leg{(i // 2) + 1}",
            "side": "left" if i % 2 == 0 else "right",
            "pair_number": (i // 2) + 1,
            "attachment_point": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},
            "upper_segment": {
                "geometry": {"type": "cylinder",
                             "radius": genes[idx],
                             "length": genes[idx + 1]},
                "inertial": {"mass": genes[idx + 2],
                             "inertia": {"ixx": 0.001, "iyy": 0.001, "izz": 0.001}},
                "joint": {"type": "revolute", "axis": [0, 1, 0],
                          "limits": {"lower":    genes[idx + 3],
                                     "upper":    genes[idx + 4],
                                     "effort":   genes[idx + 5],
                                     "velocity": genes[idx + 6]},
                          "stiffness": genes[idx + 7]},
                "visual": {"color": {"rgba": [0.8, 0.4, 0.2, 1.0]}}  # fixed
            },
            "lower_segment": {
                "geometry": {"type": "cylinder",
                             "radius": genes[idx + 8],
                             "length": genes[idx + 9]},
                "inertial": {"mass": genes[idx + 10],
                             "inertia": {"ixx": 0.001, "iyy": 0.001, "izz": 0.001}},
                "joint": {"type": "revolute", "axis": [0, 1, 0],
                          "limits": {"lower":    genes[idx + 11],
                                     "upper":    genes[idx + 12],
                                     "effort":   50.0,   # fixed defaults
                                     "velocity": 2.0},
                          "stiffness": genes[idx + 13]},
                "visual": {"color": {"rgba": [0.5, 0.5, 0.5, 1.0]},
                           "origin_offset": {"xyz": [0, 0, -genes[idx + 9] / 2]}}
            },
            "foot": {
                "geometry": {"type": "box",
                             "size": {"x": 0.4, "y": 0.2, "z": 0.1}},  # fixed size
                "inertial": {"mass": 1.0,
                             "inertia": {"ixx": 0.001, "iyy": 0.001, "izz": 0.001}},
                "joint": {"type": "revolute", "axis": [0, 1, 0],
                          "limits": {"lower": -1.0, "upper": 1.0,
                                     "effort": 50, "velocity": 1.0},
                          "stiffness": 1.0},
                "visual": {"color": {"rgba": [0.3, 0.3, 0.3, 1.0]},
                           "origin_offset": {"xyz": [0, 0, -genes[idx + 9] / 2]}}
            }
        }
        genome["leg_params"].append(leg)

    positions = _calculate_leg_positions(NUM_LEGS, genome["base_body"]["geometry"]["size"])
    for i, pos in enumerate(positions):
        genome["leg_params"][i]["attachment_point"]["xyz"] = pos

    return genome


def generate_random_creature_genome(
        min_legs: int = 2,
        max_legs: int = 8,
        seed: int = None
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    possible_legs = [i for i in range(min_legs, max_legs + 1, 2) if i % 2 == 0]
    num_legs = random.choice(possible_legs)

    genome = {
        "creature_name": f"creature_{random.randint(1000, 9999)}",
        "global_params": {
            "num_legs": num_legs,
            "symmetry": "bilateral"
        },
        "base_body": {
            "geometry": {
                "type": "box",
                "size": {
                    "x": random.uniform(1.5, 3.0),
                    "y": random.uniform(0.6, 1.2),
                    "z": random.uniform(0.4, 0.8)
                }
            },
            "inertial": {
                "mass": random.uniform(8.0, 15.0),
                "inertia": {
                    "ixx": random.uniform(0.001, 0.01),
                    "iyy": random.uniform(0.001, 0.01),
                    "izz": random.uniform(0.001, 0.01)
                }
            },
            "visual": {"color": {"rgba": [0.4, 0.6, 0.8, 1.0]}},
            "base_height": random.uniform(1.5, 2.5)
        },
        "leg_params": []
    }

    leg_positions = _calculate_leg_positions(num_legs, genome["base_body"]["geometry"]["size"])

    for i in range(num_legs):
        side = "left" if i % 2 == 0 else "right"
        pair_num = (i // 2) + 1

        leg = {
            "leg_id": i,
            "leg_name": f"{side}Leg{pair_num}",
            "side": side,
            "pair_number": pair_num,
            "attachment_point": {"xyz": leg_positions[i], "rpy": [0, 0, 0]},
            "upper_segment": {
                "geometry": {
                    "type": "cylinder",
                    "radius": random.uniform(0.1, 0.2),
                    "length": random.uniform(0.6, 1.0)
                },
                "inertial": {"mass": random.uniform(0.5, 2.0),
                             "inertia": {"ixx": 0.001, "iyy": 0.001, "izz": 0.001}},
                "visual": {"color": {"rgba": [0.8, 0.4, 0.2, 1.0]},
                           "origin_offset": {"xyz": [0, 0, None]}},
                "joint": {
                    "type": "revolute", "axis": [0, 1, 0],
                    "limits": {
                        "lower": random.uniform(-3.14, -1.57),
                        "upper": random.uniform(1.57, 3.14),
                        "effort": random.uniform(50.0, 150.0),
                        "velocity": random.uniform(1.0, 3.0)
                    },
                    "stiffness": random.uniform(0.1, 2.0)
                }
            },
            "lower_segment": {
                "geometry": {
                    "type": "cylinder",
                    "radius": random.uniform(0.08, 0.15),
                    "length": random.uniform(0.8, 1.3)
                },
                "inertial": {"mass": random.uniform(0.5, 1.5),
                             "inertia": {"ixx": 0.001, "iyy": 0.001, "izz": 0.001}},
                "visual": {"color": {"rgba": [0.5, 0.5, 0.5, 1.0]},
                           "origin_offset": {"xyz": [0, 0, None]}},
                "joint": {
                    "type": "revolute", "axis": [0, 1, 0],
                    "limits": {
                        "lower": random.uniform(-3.14, -1.0),
                        "upper": random.uniform(1.0, 3.14),
                        "effort": random.uniform(50.0, 150.0),
                        "velocity": random.uniform(1.0, 3.0)
                    },
                    "stiffness": random.uniform(0.1, 2.0)
                },
                "joint_offset": None
            },
            "foot": {
                "geometry": {
                    "type": "box",
                    "size": {"x": 0.4, "y": 0.2, "z": 0.1}
                },
                "inertial": {"mass": random.uniform(0.5, 1.5),
                             "inertia": {"ixx": 0.001, "iyy": 0.001, "izz": 0.001}},
                "visual": {"color": {"rgba": [0.3, 0.3, 0.3, 1.0]},
                           "origin_offset": {"xyz": [None, 0, 0]}},
                "joint": {
                    "type": "revolute", "axis": [0, 1, 0],
                    "limits": {
                        "lower": random.uniform(-3.14, -0.5),
                        "upper": random.uniform(0.5, 3.14),
                        "effort": random.uniform(50.0, 150.0),
                        "velocity": random.uniform(1.0, 3.0)
                    },
                    "stiffness": random.uniform(0.1, 2.0)
                },
                "joint_offset": None
            }
        }
        genome["leg_params"].append(leg)

    return genome


def _calculate_leg_positions(num_legs: int, body_size: Dict[str, float]) -> List[List[float]]:
    positions = []
    pairs = num_legs // 2

    if pairs == 1:
        x_positions = [0]
    else:
        x_spacing = body_size["x"] * 0.7 / (pairs - 1)
        x_start = body_size["x"] * 0.35
        x_positions = [x_start - i * x_spacing for i in range(pairs)]

    y_offset = body_size["y"] * 0.75
    z_offset = 0

    for x_pos in x_positions:
        positions.append([x_pos,  y_offset, z_offset])
        positions.append([x_pos, -y_offset, z_offset])

    return positions


def genome_to_urdf(genome: Dict[str, Any], output_file: str = None) -> str:

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{genome["creature_name"]}">')

    lines.append('  <link name="base_footprint">')
    lines.append('    <inertial>')
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    lines.append('      <mass value="0.001" />')
    lines.append('      <inertia ixx="0.0001" ixy="0" ixz="0" '
                 'iyy="0.0001" iyz="0" izz="0.0001" />')
    lines.append('    </inertial>')
    lines.append('  </link>')

    base_height = genome["base_body"]["base_height"]
    lines.append('  <joint name="base_joint" type="fixed">')
    lines.append('    <parent link="base_footprint" />')
    lines.append('    <child link="base_link" />')
    lines.append(f'    <origin xyz="0 0 {base_height}" rpy="0 0 0" />')
    lines.append('  </joint>')

    lines.append('  <link name="base_link">')
    body = genome["base_body"]

    lines.append('    <visual>')
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    lines.append('      <geometry>')
    if body["geometry"]["type"] == "box":
        size = body["geometry"]["size"]
        lines.append(f'        <box size="{size["x"]} {size["y"]} {size["z"]}" />')
    lines.append('      </geometry>')
    color = body["visual"]["color"]["rgba"]
    lines.append('      <material name="base_link-material">')
    lines.append(f'        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}" />')
    lines.append('      </material>')
    lines.append('    </visual>')

    lines.append('    <collision>')
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    lines.append('      <geometry>')
    if body["geometry"]["type"] == "box":
        size = body["geometry"]["size"]
        lines.append(f'        <box size="{size["x"]} {size["y"]} {size["z"]}" />')
    lines.append('      </geometry>')
    lines.append('    </collision>')

    lines.append('    <inertial>')
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    mass = body["inertial"]["mass"]
    lines.append(f'      <mass value="{mass}" />')
    inertia = body["inertial"]["inertia"]
    lines.append(f'      <inertia ixx="{inertia["ixx"]}" ixy="0" ixz="0" '
                 f'iyy="{inertia["iyy"]}" iyz="0" izz="{inertia["izz"]}" />')
    lines.append('    </inertial>')
    lines.append('  </link>')
    lines.append('')

    for leg in genome["leg_params"]:
        leg_name = leg["leg_name"]

        upper_len   = leg["upper_segment"]["geometry"]["length"]
        lower_len   = leg["lower_segment"]["geometry"]["length"]
        foot_size_x = leg["foot"]["geometry"]["size"]["x"]

        upper_visual_z   = -upper_len / 2
        lower_visual_z   = -lower_len / 2
        foot_visual_x    = foot_size_x / 2
        upper_to_lower_z = -(upper_len + 0.05)
        lower_to_foot_z  = -lower_len

        # ── Upper leg ────────────────────────────────────────────────────────
        upper_joint_name = f"base_link_to_upper{leg_name}"
        upper_link_name  = f"upper{leg_name}"

        lines.append(f'  <joint name="{upper_joint_name}" type="revolute">')
        lines.append('    <parent link="base_link" />')
        lines.append(f'    <child link="{upper_link_name}" />')
        pos = leg["attachment_point"]["xyz"]
        lines.append(f'    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="0 0 0" />')
        axis = leg["upper_segment"]["joint"]["axis"]
        lines.append(f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>')
        limits = leg["upper_segment"]["joint"]["limits"]
        lines.append(f'    <limit effort="{limits["effort"]}" lower="{limits["lower"]}" '
                     f'upper="{limits["upper"]}" velocity="{limits["velocity"]}"/>')
        lines.append('  </joint>')

        lines.append(f'  <link name="{upper_link_name}">')
        lines.append('    <visual>')
        lines.append(f'      <origin xyz="0 0 {upper_visual_z}" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <cylinder radius="{leg["upper_segment"]["geometry"]["radius"]}" '
                     f'length="{upper_len}" />')
        lines.append('      </geometry>')
        color = leg["upper_segment"]["visual"]["color"]["rgba"]
        lines.append(f'      <material name="{upper_link_name}-material">')
        lines.append(f'        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}" />')
        lines.append('      </material>')
        lines.append('    </visual>')
        lines.append('    <collision>')
        lines.append(f'      <origin xyz="0 0 {upper_visual_z}" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <cylinder radius="{leg["upper_segment"]["geometry"]["radius"]}" '
                     f'length="{upper_len}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')
        lines.append('    <inertial>')
        lines.append(f'      <origin xyz="0 0 {upper_visual_z}" rpy="0 0 0" />')
        mass = leg["upper_segment"]["inertial"]["mass"]
        lines.append(f'      <mass value="{mass}" />')
        inertia = leg["upper_segment"]["inertial"]["inertia"]
        lines.append(f'      <inertia ixx="{inertia["ixx"]}" ixy="0" ixz="0" '
                     f'iyy="{inertia["iyy"]}" iyz="0" izz="{inertia["izz"]}" />')
        lines.append('    </inertial>')
        lines.append('  </link>')
        lines.append('')

        # ── Lower leg ────────────────────────────────────────────────────────
        lower_joint_name = f"upper{leg_name}_to_lower{leg_name}"
        lower_link_name  = f"lower{leg_name}"

        lines.append(f'  <joint name="{lower_joint_name}" type="revolute">')
        lines.append(f'    <parent link="{upper_link_name}" />')
        lines.append(f'    <child link="{lower_link_name}" />')
        lines.append(f'    <origin xyz="0 0 {upper_to_lower_z}" rpy="0 0 0" />')
        axis = leg["lower_segment"]["joint"]["axis"]
        lines.append(f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>')
        limits = leg["lower_segment"]["joint"]["limits"]
        lines.append(f'    <limit effort="{limits["effort"]}" lower="{limits["lower"]}" '
                     f'upper="{limits["upper"]}" velocity="{limits["velocity"]}"/>')
        lines.append('  </joint>')

        lines.append(f'  <link name="{lower_link_name}">')
        lines.append('    <visual>')
        lines.append(f'      <origin xyz="0 0 {lower_visual_z}" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <cylinder radius="{leg["lower_segment"]["geometry"]["radius"]}" '
                     f'length="{lower_len}" />')
        lines.append('      </geometry>')
        color = leg["lower_segment"]["visual"]["color"]["rgba"]
        lines.append(f'      <material name="{lower_link_name}-material">')
        lines.append(f'        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}" />')
        lines.append('      </material>')
        lines.append('    </visual>')
        lines.append('    <collision>')
        lines.append(f'      <origin xyz="0 0 {lower_visual_z}" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <cylinder radius="{leg["lower_segment"]["geometry"]["radius"]}" '
                     f'length="{lower_len}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')
        lines.append('    <inertial>')
        lines.append(f'      <origin xyz="0 0 {lower_visual_z}" rpy="0 0 0" />')
        mass = leg["lower_segment"]["inertial"]["mass"]
        lines.append(f'      <mass value="{mass}" />')
        inertia = leg["lower_segment"]["inertial"]["inertia"]
        lines.append(f'      <inertia ixx="{inertia["ixx"]}" ixy="0" ixz="0" '
                     f'iyy="{inertia["iyy"]}" iyz="0" izz="{inertia["izz"]}" />')
        lines.append('    </inertial>')
        lines.append('  </link>')
        lines.append('')

        # ── Foot ─────────────────────────────────────────────────────────────
        foot_joint_name = f"lower{leg_name}_to_{leg['side']}Foot{leg['pair_number']}"
        foot_link_name  = f"{leg['side']}Foot{leg['pair_number']}"

        lines.append(f'  <joint name="{foot_joint_name}" type="revolute">')
        lines.append(f'    <parent link="{lower_link_name}" />')
        lines.append(f'    <child link="{foot_link_name}" />')
        lines.append(f'    <origin xyz="0 0 {lower_to_foot_z}" rpy="0 0 0" />')
        axis = leg["foot"]["joint"]["axis"]
        lines.append(f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>')
        limits = leg["foot"]["joint"]["limits"]
        lines.append(f'    <limit effort="{limits["effort"]}" lower="{limits["lower"]}" '
                     f'upper="{limits["upper"]}" velocity="{limits["velocity"]}"/>')
        lines.append('  </joint>')

        lines.append(f'  <link name="{foot_link_name}">')
        lines.append('    <visual>')
        lines.append(f'      <origin xyz="{foot_visual_x} 0 0" rpy="0 0 0" />')
        lines.append('      <geometry>')
        foot_size = leg["foot"]["geometry"]["size"]
        lines.append(f'        <box size="{foot_size["x"]} {foot_size["y"]} {foot_size["z"]}" />')
        lines.append('      </geometry>')
        color = leg["foot"]["visual"]["color"]["rgba"]
        lines.append(f'      <material name="{foot_link_name}-material">')
        lines.append(f'        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}" />')
        lines.append('      </material>')
        lines.append('    </visual>')
        lines.append('    <collision>')
        lines.append(f'      <origin xyz="{foot_visual_x} 0 0" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <box size="{foot_size["x"]} {foot_size["y"]} {foot_size["z"]}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')
        lines.append('    <inertial>')
        lines.append(f'      <origin xyz="{foot_visual_x} 0 0" rpy="0 0 0" />')
        mass = leg["foot"]["inertial"]["mass"]
        lines.append(f'      <mass value="{mass}" />')
        inertia = leg["foot"]["inertial"]["inertia"]
        lines.append(f'      <inertia ixx="{inertia["ixx"]}" ixy="0" ixz="0" '
                     f'iyy="{inertia["iyy"]}" iyz="0" izz="{inertia["izz"]}" />')
        lines.append('    </inertial>')
        lines.append('  </link>')
        lines.append('')

    lines.append('</robot>')
    urdf_string = '\n'.join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(urdf_string)
        print(f"URDF saved to {output_file}")

    return urdf_string


def save_genome_to_json(genome: Dict[str, Any], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(genome, f, indent=2)
    print(f"Genome saved to {filename}")


def load_genome_from_json(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as f:
        return json.load(f)