from typing import Dict, Any

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

