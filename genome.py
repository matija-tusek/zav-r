#inicijalna struktura za GA
#input: fitness score iz RL, velicina populacije, ostale postavke GA....
#potrebno i definirati raspon za vrijednosti pojedinih parametara (globalnih)
#output: nove generacije, statistika po generacijama

'''
Globalni parametri (geni):
-> broj nogu (2,4,6,8)
-> simetrija nogu (kako i gdje su postavljene na biće)
-> oblik noge (duljina pojedinih dijelova noge, velicina stopala, broj zglobova...)
-> oblik gornjeg dijela tijela (duljina, širina, visina i oblik)
-> masa pojedinih dijelova tijela
-> raspon okretanja pojedinih zglobova (npr. kuk, koljeno, gležanj..)
-> krutost zglobova (koliko se lako okreću)
'''

import math
import random
import pygad
import numpy as np



import json
import random
from typing import Dict, List, Any




def genome_from_genes(genes,NUM_LEGS):
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
            "visual": {"color": {"rgba": [genes[7], genes[8], genes[9], 1.0]}},
            "base_height": 1.5
        },
        "leg_params": []
    }

    # Starting index for leg genes
    leg_gene_start = 10
    leg_genes_per_leg = 20  # adjust depending on how many leg params you encode

    for i in range(NUM_LEGS):
        idx = leg_gene_start + i * leg_genes_per_leg
        leg = {
            "leg_id": i,
            "leg_name": f"{'left' if i % 2 == 0 else 'right'}Leg{(i // 2) + 1}",
            "side": "left" if i%2==0 else "right",
            "pair_number": (i//2)+1,
            "attachment_point": {"xyz": [0,0,0], "rpy": [0,0,0]},  # will calculate after
            "upper_segment": {
                "geometry": {"type": "cylinder", "radius": genes[idx], "length": genes[idx+1]},
                "inertial": {"mass": genes[idx+2], "inertia": {"ixx":0.001,"iyy":0.001,"izz":0.001}},
                "joint": {"type":"revolute","axis":[0,1,0],
                          "limits":{"lower":genes[idx+3],"upper":genes[idx+4],
                                    "effort":genes[idx+5],"velocity":genes[idx+6]},
                          "stiffness":genes[idx+7]},
                "visual": {"color":{"rgba":[genes[idx+8], genes[idx+9], genes[idx+10],1.0]}}
            },
            "lower_segment": {
                "geometry": {"type": "cylinder", "radius": genes[idx+11], "length": genes[idx+12]},
                "inertial": {"mass": genes[idx+13], "inertia": {"ixx":0.001,"iyy":0.001,"izz":0.001}},
                "joint": {"type":"revolute","axis":[0,1,0],
                          "limits":{"lower":genes[idx+14],"upper":genes[idx+15],
                                    "effort":genes[idx+16],"velocity":genes[idx+17]},
                          "stiffness":genes[idx+18]},
                "visual": {
                    "color": {"rgba": [0.5, 0.5, 0.5, 1.0]},  # default gray or map from genes if you have extra
                    "origin_offset": {"xyz": [0, 0, -genes[idx + 12] / 2]}  # optional
                }
            },
            "foot": {
                "geometry": {"type":"box","size":{"x":genes[idx+19],"y":0.2,"z":0.1}},  # example
                "inertial":{"mass":1.0,"inertia":{"ixx":0.001,"iyy":0.001,"izz":0.001}},
                "joint":{"type":"revolute","axis":[0,1,0],"limits":{"lower":-1.0,"upper":1.0,"effort":50,"velocity":1.0},
                        "stiffness":1.0},
                "visual": {
                    "color": {"rgba": [0.5, 0.5, 0.5, 1.0]},  # default gray or map from genes if you have extra
                    "origin_offset": {"xyz": [0, 0, -genes[idx + 12] / 2]}  # optional
                }
            }
        }
        genome["leg_params"].append(leg)

    # Calculate attachment points based on body geometry
    positions = _calculate_leg_positions(NUM_LEGS, genome["base_body"]["geometry"]["size"])
    for i,pos in enumerate(positions):
        genome["leg_params"][i]["attachment_point"]["xyz"] = pos

    return genome


def generate_random_creature_genome(
        min_legs: int = 2,
        max_legs: int = 8,
        seed: int = None
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    possible_legs = [i for i in range(min_legs, max_legs + 1, 2) if i % 2 == 0] #paran broj
    num_legs = random.choice(possible_legs)

    genome = {
        "creature_name": f"creature_{random.randint(1000, 9999)}",

        # Global parameters
        "global_params": {
            "num_legs": num_legs,
            "symmetry": "bilateral"  #  bilateral, radial, asymmetric
        },

        # Base body parameters
        "base_body": {
            "geometry": {
                "type": "box",  # box, cylinder, sphere
                "size": {
                    "x": random.uniform(1.5, 3.0),  # length
                    "y": random.uniform(0.6, 1.2),  # width
                    "z": random.uniform(0.4, 0.8)  # height
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
            "visual": {
                "color": {
                    "rgba": [random.random(), random.random(), random.random(), 1.0]
                }
            },
            "base_height": random.uniform(1.5, 2.5)  # Height above ground (base_joint z)
        },

        # Leg configuration
        "leg_params": []
    }

    # Generate parameters for each leg
    leg_positions = _calculate_leg_positions(num_legs, genome["base_body"]["geometry"]["size"])

    for i in range(num_legs):
        side = "left" if i % 2 == 0 else "right"
        pair_num = (i // 2) + 1

        leg = {
            "leg_id": i,
            "leg_name": f"{side}Leg{pair_num}",
            "side": side,
            "pair_number": pair_num,

            # Position on body (joint origin from base_link)
            "attachment_point": {
                "xyz": leg_positions[i],
                "rpy": [0, 0, 0]
            },

            # Upper leg segment (thigh)
            "upper_segment": {
                "geometry": {
                    "type": "cylinder",
                    "radius": random.uniform(0.1, 0.2),
                    "length": random.uniform(0.6, 1.0)
                },
                "inertial": {
                    "mass": random.uniform(0.5, 2.0),
                    "inertia": {
                        "ixx": 0.001,
                        "iyy": 0.001,
                        "izz": 0.001
                    }
                },
                "visual": {
                    "color": {
                        "rgba": [random.random(), random.random(), random.random(), 1.0]
                    },
                    "origin_offset": {  # Visual origin relative to joint
                        "xyz": [0, 0, None]  # Will be calculated as -length/2
                    }
                },
                "joint": {
                    "type": "revolute",
                    "axis": [0, 1, 0],  # Y-axis rotation (pitch)
                    "limits": {
                        "lower": random.uniform(-3.14, -1.57),
                        "upper": random.uniform(1.57, 3.14),
                        "effort": random.uniform(50.0, 150.0),
                        "velocity": random.uniform(1.0, 3.0)
                    },
                    "stiffness": random.uniform(0.1, 2.0)  # Joint stiffness/damping
                }
            },

            # Lower leg segment (shin)
            "lower_segment": {
                "geometry": {
                    "type": "cylinder",
                    "radius": random.uniform(0.08, 0.15),
                    "length": random.uniform(0.8, 1.3)
                },
                "inertial": {
                    "mass": random.uniform(0.5, 1.5),
                    "inertia": {
                        "ixx": 0.001,
                        "iyy": 0.001,
                        "izz": 0.001
                    }
                },
                "visual": {
                    "color": {
                        "rgba": [random.random(), random.random(), random.random(), 1.0]
                    },
                    "origin_offset": {
                        "xyz": [0, 0, None]  # Will be calculated as -length/2
                    }
                },
                "joint": {
                    "type": "revolute",
                    "axis": [0, 1, 0],
                    "limits": {
                        "lower": random.uniform(-3.14, -1.0),
                        "upper": random.uniform(1.0, 3.14),
                        "effort": random.uniform(50.0, 150.0),
                        "velocity": random.uniform(1.0, 3.0)
                    },
                    "stiffness": random.uniform(0.1, 2.0)
                },
                "joint_offset": None  # Will be calculated as -(upper_length + small_gap)
            },

            # Foot
            "foot": {
                "geometry": {
                    "type": "box",
                    "size": {
                        "x": random.uniform(0.3, 0.6),  # length (forward)
                        "y": random.uniform(0.15, 0.25),  # width
                        "z": random.uniform(0.08, 0.15)  # height
                    }
                },
                "inertial": {
                    "mass": random.uniform(0.5, 1.5),
                    "inertia": {
                        "ixx": 0.001,
                        "iyy": 0.001,
                        "izz": 0.001
                    }
                },
                "visual": {
                    "color": {
                        "rgba": [random.random(), random.random(), random.random(), 1.0]
                    },
                    "origin_offset": {  # Foot extends forward
                        "xyz": [None, 0, 0]  # x will be calculated as size_x/2
                    }
                },
                "joint": {
                    "type": "revolute",
                    "axis": [0, 1, 0],
                    "limits": {
                        "lower": random.uniform(-3.14, -0.5),
                        "upper": random.uniform(0.5, 3.14),
                        "effort": random.uniform(50.0, 150.0),
                        "velocity": random.uniform(1.0, 3.0)
                    },
                    "stiffness": random.uniform(0.1, 2.0)
                },
                "joint_offset": None  # Will be calculated as -lower_length
            }
        }

        genome["leg_params"].append(leg)

    return genome


def _calculate_leg_positions(num_legs: int, body_size: Dict[str, float]) -> List[List[float]]:
    """
    Calculate attachment points for legs on the body based on bilateral symmetry.
    """
    positions = []
    pairs = num_legs // 2

    # Distribute pairs along the length of the body
    if pairs == 1:
        x_positions = [0]
    else:
        x_spacing = body_size["x"] * 0.7 / (pairs - 1)  # 70% of body length
        x_start = body_size["x"] * 0.35
        x_positions = [x_start - i * x_spacing for i in range(pairs)]

    # Y position (width) - legs attach to sides
    y_offset = body_size["y"] * 0.75  # 75% of half-width

    # Z position (height) - legs attach at body center height
    z_offset = 0

    # Create positions: alternating left-right for each pair
    for x_pos in x_positions:
        positions.append([x_pos, y_offset, z_offset])  # Left leg
        positions.append([x_pos, -y_offset, z_offset])  # Right leg

    return positions


def genome_to_urdf(genome: Dict[str, Any], output_file: str = None) -> str:

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{genome["creature_name"]}">')

    # Base footprint (ground reference)
    lines.append('  <link name="base_footprint">')
    lines.append('    <inertial>')
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    lines.append('      <mass value="0.001" />')
    lines.append('      <inertia ixx="0.0001" ixy="0" ixz="0" '
                 'iyy="0.0001" iyz="0" izz="0.0001" />')
    lines.append('    </inertial>')
    lines.append('  </link>')

    # Base joint (connects footprint to main body)
    base_height = genome["base_body"]["base_height"]
    lines.append('  <joint name="base_joint" type="fixed">')
    lines.append('    <parent link="base_footprint" />')
    lines.append('    <child link="base_link" />')
    lines.append(f'    <origin xyz="0 0 {base_height}" rpy="0 0 0" />')
    lines.append('  </joint>')

    # Base link (main body)
    lines.append('  <link name="base_link">')
    body = genome["base_body"]

    # Visual
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

    # Collision
    lines.append('    <collision>')
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    lines.append('      <geometry>')
    if body["geometry"]["type"] == "box":
        size = body["geometry"]["size"]
        lines.append(f'        <box size="{size["x"]} {size["y"]} {size["z"]}" />')
    lines.append('      </geometry>')
    lines.append('    </collision>')

    # Inertial
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

    # Generate legs
    for leg in genome["leg_params"]:
        leg_name = leg["leg_name"]

        # Calculate offsets for visual/collision origins
        upper_len = leg["upper_segment"]["geometry"]["length"]
        lower_len = leg["lower_segment"]["geometry"]["length"]
        foot_size_x = leg["foot"]["geometry"]["size"]["x"]

        upper_visual_z = -upper_len / 2
        lower_visual_z = -lower_len / 2
        foot_visual_x = foot_size_x / 2

        # Joint offset calculations
        upper_to_lower_z = -(upper_len + 0.05)  # small gap
        lower_to_foot_z = -lower_len

        # ===== UPPER LEG =====
        upper_joint_name = f"base_link_to_upper{leg_name}"
        upper_link_name = f"upper{leg_name}"

        # Upper leg joint
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

        # Upper leg link
        lines.append(f'  <link name="{upper_link_name}">')

        # Visual
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

        # Collision
        lines.append('    <collision>')
        lines.append(f'      <origin xyz="0 0 {upper_visual_z}" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <cylinder radius="{leg["upper_segment"]["geometry"]["radius"]}" '
                     f'length="{upper_len}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')

        # Inertial
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

        # ===== LOWER LEG =====
        lower_joint_name = f"upper{leg_name}_to_lower{leg_name}"
        lower_link_name = f"lower{leg_name}"

        # Lower leg joint
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

        # Lower leg link
        lines.append(f'  <link name="{lower_link_name}">')

        # Visual
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

        # Collision
        lines.append('    <collision>')
        lines.append(f'      <origin xyz="0 0 {lower_visual_z}" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <cylinder radius="{leg["lower_segment"]["geometry"]["radius"]}" '
                     f'length="{lower_len}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')

        # Inertial
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

        # ===== FOOT =====
        foot_joint_name = f"lower{leg_name}_to_{leg['side']}Foot{leg['pair_number']}"
        foot_link_name = f"{leg['side']}Foot{leg['pair_number']}"

        # Foot joint
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

        # Foot link
        lines.append(f'  <link name="{foot_link_name}">')

        # Visual
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

        # Collision
        lines.append('    <collision>')
        lines.append(f'      <origin xyz="{foot_visual_x} 0 0" rpy="0 0 0" />')
        lines.append('      <geometry>')
        lines.append(f'        <box size="{foot_size["x"]} {foot_size["y"]} {foot_size["z"]}" />')
        lines.append('      </geometry>')
        lines.append('    </collision>')

        # Inertial
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

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(urdf_string)
        print(f"URDF saved to {output_file}")

    return urdf_string

def save_genome_to_json(genome: Dict[str, Any], filename: str ) -> None:
    """Save the genome to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(genome, f, indent=2)
    print(f"Genome saved to {filename}")


def load_genome_from_json(filename: str) -> Dict[str, Any]:
    """Load a genome from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)




