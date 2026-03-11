import math
import random
import numpy as np
import json
from typing import Dict, List, Any

# ── Gene layout (per leg, 11 genes) ───────────────────────────────────────────
#
#  BODY  (4 genes, indices 0-3)
#   0  body_x          length of torso
#   1  body_y          width  of torso
#   2  body_z          height of torso
#   3  body_mass       mass of torso
#                      ixx/iyy/izz calculated automatically from geometry
#
#  PER LEG  (11 genes, starting at index 4)
#   idx+0   upper_radius        thickness of upper segment
#   idx+1   total_leg_length    total leg length (upper+lower combined)
#   idx+2   upper_ratio         fraction of total that is upper  (0.3-0.7)
#   idx+3   upper_mass
#   idx+4   upper_joint_lower   hip joint lower limit  (rad)
#   idx+5   upper_joint_upper   hip joint upper limit  (rad)
#   idx+6   upper_stiffness
#   idx+7   lower_radius
#   idx+8   lower_mass
#   idx+9   lower_joint_lower   knee joint lower limit (rad)
#   idx+10  foot_length         length of foot box
#           lower_stiffness     fixed at 1.5
#           effort              calculated from mass
#           velocity            fixed at 2.0
#           hip_angle_z         from attachment position calculation
#
# TOTAL: 4 + NUM_LEGS * 11

BODY_GENES    = 4
LEG_GENES     = 11
FIXED_EFFORT_PER_KG = 30.0   # effort = mass * this constant
FIXED_VELOCITY      = 2.0
FIXED_LOWER_STIFFNESS = 1.5


def _auto_inertia(mass: float, sx: float, sy: float, sz: float) -> dict:
    """Box inertia tensor from mass and dimensions."""
    return {
        "ixx": round(mass / 12.0 * (sy**2 + sz**2), 6),
        "iyy": round(mass / 12.0 * (sx**2 + sz**2), 6),
        "izz": round(mass / 12.0 * (sx**2 + sy**2), 6),
    }


def _cylinder_inertia(mass: float, r: float, length: float) -> dict:
    """Cylinder inertia tensor."""
    ixx = mass / 12.0 * (3 * r**2 + length**2)
    izz = mass / 2.0  * r**2
    return {"ixx": round(ixx, 6), "iyy": round(ixx, 6), "izz": round(izz, 6)}


def genome_from_genes(genes, NUM_LEGS: int) -> dict:
    # ── Body ──────────────────────────────────────────────────────────────────
    body_x    = float(genes[0])
    body_y    = float(genes[1])
    body_z    = float(genes[2])
    body_mass = float(genes[3])

    genome = {
        "creature_name": f"creature_{np.random.randint(1000, 9999)}",
        "global_params": {"num_legs": NUM_LEGS, "symmetry": "bilateral"},
        "base_body": {
            "geometry":  {"type": "box", "size": {"x": body_x, "y": body_y, "z": body_z}},
            "inertial":  {"mass": body_mass,
                          "inertia": _auto_inertia(body_mass, body_x, body_y, body_z)},
            "visual":    {"color": {"rgba": [0.4, 0.6, 0.8, 1.0]}},
            "base_height": 1.5,
        },
        "leg_params": [],
    }

    # ── Legs ──────────────────────────────────────────────────────────────────
    for i in range(NUM_LEGS):
        idx = BODY_GENES + i * LEG_GENES

        upper_radius      = float(genes[idx + 0])
        total_leg_length  = float(genes[idx + 1])
        upper_ratio       = float(np.clip(genes[idx + 2], 0.3, 0.7))
        upper_mass        = float(genes[idx + 3])
        upper_jnt_lower   = float(genes[idx + 4])
        upper_jnt_upper   = float(genes[idx + 5])
        upper_stiffness   = float(genes[idx + 6])
        lower_radius      = float(genes[idx + 7])
        lower_mass        = float(genes[idx + 8])
        lower_jnt_lower   = float(genes[idx + 9])
        foot_length       = float(genes[idx + 10])

        # Derived geometry
        upper_length = total_leg_length * upper_ratio
        lower_length = total_leg_length * (1.0 - upper_ratio)

        # Effort scales with mass so heavier limbs are still controllable
        upper_effort = upper_mass * FIXED_EFFORT_PER_KG
        lower_effort = lower_mass * FIXED_EFFORT_PER_KG

        # Knee only bends backwards (lower > 0 not meaningful for walking)
        lower_jnt_upper = 0.1

        leg = {
            "leg_id":     i,
            "leg_name":   f"{'left' if i % 2 == 0 else 'right'}Leg{(i // 2) + 1}",
            "side":       "left" if i % 2 == 0 else "right",
            "pair_number": (i // 2) + 1,
            "attachment_point": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]},

            "upper_segment": {
                "geometry": {"type": "cylinder",
                             "radius": upper_radius,
                             "length": upper_length},
                "inertial": {"mass": upper_mass,
                             "inertia": _cylinder_inertia(upper_mass, upper_radius, upper_length)},
                "joint": {
                    "type": "revolute", "axis": [0, 1, 0],
                    "limits": {
                        "lower":    upper_jnt_lower,
                        "upper":    upper_jnt_upper,
                        "effort":   upper_effort,
                        "velocity": FIXED_VELOCITY,
                    },
                    "stiffness": upper_stiffness,
                },
                "visual": {"color": {"rgba": [0.8, 0.4, 0.2, 1.0]}},
            },

            "lower_segment": {
                "geometry": {"type": "cylinder",
                             "radius": lower_radius,
                             "length": lower_length},
                "inertial": {"mass": lower_mass,
                             "inertia": _cylinder_inertia(lower_mass, lower_radius, lower_length)},
                "joint": {
                    "type": "revolute", "axis": [0, 1, 0],
                    "limits": {
                        "lower":    lower_jnt_lower,
                        "upper":    lower_jnt_upper,
                        "effort":   lower_effort,
                        "velocity": FIXED_VELOCITY,
                    },
                    "stiffness": FIXED_LOWER_STIFFNESS,
                },
                "visual": {"color": {"rgba": [0.5, 0.5, 0.5, 1.0]},
                           "origin_offset": {"xyz": [0, 0, -lower_length / 2]}},
            },

            "foot": {
                "geometry": {"type": "box",
                             "size": {"x": foot_length, "y": 0.18, "z": 0.08}},
                "inertial": {"mass": 0.3,
                             "inertia": _auto_inertia(0.3, foot_length, 0.18, 0.08)},
                "joint": {
                    "type": "revolute", "axis": [0, 1, 0],
                    "limits": {"lower": -0.5, "upper": 0.5,
                               "effort": 20.0, "velocity": 1.5},
                    "stiffness": 1.0,
                },
                "visual": {"color": {"rgba": [0.25, 0.25, 0.25, 1.0]},
                           "origin_offset": {"xyz": [foot_length / 2, 0, 0]}},
            },
        }
        genome["leg_params"].append(leg)

    # Attachment positions (includes slight downward angle for hip)
    positions = _calculate_leg_positions(NUM_LEGS, genome["base_body"]["geometry"]["size"])
    for i, pos in enumerate(positions):
        genome["leg_params"][i]["attachment_point"]["xyz"] = pos

    return genome


def _calculate_leg_positions(num_legs: int, body_size: Dict[str, float]) -> List[List[float]]:
    """
    Distribute leg attachment points along the body.
    Legs attach slightly below the body centre (z = -body_z*0.3)
    so the hip joint angles down naturally.
    """
    positions = []
    pairs = num_legs // 2

    if pairs == 1:
        x_positions = [0.0]
    else:
        span    = body_size["x"] * 0.7
        x_start = body_size["x"] * 0.35
        x_positions = [x_start - i * (span / (pairs - 1)) for i in range(pairs)]

    y_offset = body_size["y"] * 0.5   # attach at body edge, not 75%
    z_offset = -body_size["z"] * 0.3  # slightly below centre → natural downward angle

    for x_pos in x_positions:
        positions.append([x_pos,  y_offset, z_offset])   # left
        positions.append([x_pos, -y_offset, z_offset])   # right

    return positions


# ── Random genome (for testing outside GA) ────────────────────────────────────

def generate_random_creature_genome(min_legs: int = 2, max_legs: int = 8,
                                     seed: int = None) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    possible_legs = [i for i in range(min_legs, max_legs + 1, 2)]
    num_legs = random.choice(possible_legs)

    body_x    = random.uniform(0.8,  2.0)
    body_y    = random.uniform(0.5,  1.0)
    body_z    = random.uniform(0.3,  0.7)
    body_mass = random.uniform(3.0, 10.0)

    genes = [body_x, body_y, body_z, body_mass]

    for _ in range(num_legs):
        genes += [
            random.uniform(0.06, 0.18),   # upper_radius
            random.uniform(0.8,  2.0),    # total_leg_length
            random.uniform(0.35, 0.65),   # upper_ratio
            random.uniform(0.4,  1.5),    # upper_mass
            random.uniform(-1.2, -0.3),   # upper_jnt_lower
            random.uniform(0.3,   1.2),   # upper_jnt_upper
            random.uniform(0.5,   4.0),   # upper_stiffness
            random.uniform(0.05,  0.12),  # lower_radius
            random.uniform(0.3,   1.2),   # lower_mass
            random.uniform(-2.0, -0.5),   # lower_jnt_lower (knee bends back)
            random.uniform(0.15,  0.5),   # foot_length
        ]

    return genome_from_genes(genes, num_legs)


# ── URDF generation ───────────────────────────────────────────────────────────

def genome_to_urdf(genome: Dict[str, Any], output_file: str = None) -> str:

    lines = ['<?xml version="1.0"?>',
             f'<robot name="{genome["creature_name"]}">']

    # base_footprint
    lines += [
        '  <link name="base_footprint">',
        '    <inertial>',
        '      <origin xyz="0 0 0" rpy="0 0 0" />',
        '      <mass value="0.001" />',
        '      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001" />',
        '    </inertial>',
        '  </link>',
    ]

    base_h = genome["base_body"]["base_height"]
    lines += [
        '  <joint name="base_joint" type="fixed">',
        '    <parent link="base_footprint" />',
        '    <child link="base_link" />',
        f'    <origin xyz="0 0 {base_h}" rpy="0 0 0" />',
        '  </joint>',
    ]

    # base_link
    body  = genome["base_body"]
    size  = body["geometry"]["size"]
    color = body["visual"]["color"]["rgba"]
    mass  = body["inertial"]["mass"]
    iner  = body["inertial"]["inertia"]

    lines += [
        '  <link name="base_link">',
        '    <visual>',
        '      <origin xyz="0 0 0" rpy="0 0 0" />',
        '      <geometry>',
        f'        <box size="{size["x"]} {size["y"]} {size["z"]}" />',
        '      </geometry>',
        '      <material name="base_link-material">',
        f'        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}" />',
        '      </material>',
        '    </visual>',
        '    <collision>',
        '      <origin xyz="0 0 0" rpy="0 0 0" />',
        '      <geometry>',
        f'        <box size="{size["x"]} {size["y"]} {size["z"]}" />',
        '      </geometry>',
        '    </collision>',
        '    <inertial>',
        '      <origin xyz="0 0 0" rpy="0 0 0" />',
        f'      <mass value="{mass}" />',
        f'      <inertia ixx="{iner["ixx"]}" ixy="0" ixz="0" '
        f'iyy="{iner["iyy"]}" iyz="0" izz="{iner["izz"]}" />',
        '    </inertial>',
        '  </link>',
        '',
    ]

    # legs
    for leg in genome["leg_params"]:
        leg_name = leg["leg_name"]
        ul = leg["upper_segment"]["geometry"]["length"]
        ll = leg["lower_segment"]["geometry"]["length"]
        fx = leg["foot"]["geometry"]["size"]["x"]

        upper_viz_z      = -ul / 2
        lower_viz_z      = -ll / 2
        foot_viz_x       = fx / 2
        upper_to_lower_z = -(ul + 0.04)
        lower_to_foot_z  = -ll

        def _jnt(parent, child, ox, oy, oz, axis, lim):
            return [
                f'  <joint name="{parent}_to_{child}" type="revolute">',
                f'    <parent link="{parent}" />',
                f'    <child link="{child}" />',
                f'    <origin xyz="{ox} {oy} {oz}" rpy="0 0 0" />',
                f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>',
                f'    <limit effort="{lim["effort"]}" lower="{lim["lower"]}" '
                f'upper="{lim["upper"]}" velocity="{lim["velocity"]}"/>',
                '  </joint>',
            ]

        def _link(name, geom_type, geom_params, viz_xyz, mass, iner, color):
            if geom_type == "cylinder":
                geom_line = (f'        <cylinder radius="{geom_params["radius"]}" '
                             f'length="{geom_params["length"]}" />')
            else:
                s = geom_params["size"]
                geom_line = f'        <box size="{s["x"]} {s["y"]} {s["z"]}" />'
            return [
                f'  <link name="{name}">',
                '    <visual>',
                f'      <origin xyz="{viz_xyz[0]} {viz_xyz[1]} {viz_xyz[2]}" rpy="0 0 0" />',
                '      <geometry>', geom_line, '      </geometry>',
                f'      <material name="{name}-material">',
                f'        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}" />',
                '      </material>',
                '    </visual>',
                '    <collision>',
                f'      <origin xyz="{viz_xyz[0]} {viz_xyz[1]} {viz_xyz[2]}" rpy="0 0 0" />',
                '      <geometry>', geom_line, '      </geometry>',
                '    </collision>',
                '    <inertial>',
                f'      <origin xyz="{viz_xyz[0]} {viz_xyz[1]} {viz_xyz[2]}" rpy="0 0 0" />',
                f'      <mass value="{mass}" />',
                f'      <inertia ixx="{iner["ixx"]}" ixy="0" ixz="0" '
                f'iyy="{iner["iyy"]}" iyz="0" izz="{iner["izz"]}" />',
                '    </inertial>',
                '  </link>',
                '',
            ]

        # upper leg
        pos = leg["attachment_point"]["xyz"]
        upper_name = f"upper{leg_name}"
        lines += _jnt("base_link", upper_name,
                      pos[0], pos[1], pos[2],
                      leg["upper_segment"]["joint"]["axis"],
                      leg["upper_segment"]["joint"]["limits"])
        lines += _link(upper_name, "cylinder",
                       leg["upper_segment"]["geometry"],
                       [0, 0, upper_viz_z],
                       leg["upper_segment"]["inertial"]["mass"],
                       leg["upper_segment"]["inertial"]["inertia"],
                       leg["upper_segment"]["visual"]["color"]["rgba"])

        # lower leg
        lower_name = f"lower{leg_name}"
        lines += _jnt(upper_name, lower_name,
                      0, 0, upper_to_lower_z,
                      leg["lower_segment"]["joint"]["axis"],
                      leg["lower_segment"]["joint"]["limits"])
        lines += _link(lower_name, "cylinder",
                       leg["lower_segment"]["geometry"],
                       [0, 0, lower_viz_z],
                       leg["lower_segment"]["inertial"]["mass"],
                       leg["lower_segment"]["inertial"]["inertia"],
                       leg["lower_segment"]["visual"]["color"]["rgba"])

        # foot
        foot_name = f"{leg['side']}Foot{leg['pair_number']}"
        lines += _jnt(lower_name, foot_name,
                      0, 0, lower_to_foot_z,
                      leg["foot"]["joint"]["axis"],
                      leg["foot"]["joint"]["limits"])
        lines += _link(foot_name, "box",
                       leg["foot"]["geometry"],
                       [foot_viz_x, 0, 0],
                       leg["foot"]["inertial"]["mass"],
                       leg["foot"]["inertial"]["inertia"],
                       leg["foot"]["visual"]["color"]["rgba"])

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