import numpy as np
from daceypy import DA, array

from field_io import read_field_table
from plane_index import create_plane_trees
from electrostatics import no_electric_field
from tracking import euler_dz


def main():
    # Replace this path with your own field table file.
    file_path = "lens_ideal_Q.txt"

    x, y, z, bx, by, bz = read_field_table(file_path)
    z_range = (np.min(z), np.max(z))

    print("Creating plane trees...")
    plane_dict = create_plane_trees(x, y, z, bx, by, bz)
    print(f"Created trees for {len(plane_dict)} planes")

    acceleration_voltage = 100000.0
    dz = 1e-5
    z0 = 0.0
    z_end = 0.1
    z_values = np.arange(z0, z_end, dz)
    steps = len(z_values)

    potential_func = no_electric_field(acceleration_voltage)

    x0 = 0.0
    y0 = 0.0
    x0_slope = 0.0
    y0_slope = 0.0
    d0 = 0.0

    DA.init(5, 5)
    param = array([
        x0 + DA(1),
        x0_slope + DA(2),
        y0 + DA(3),
        y0_slope + DA(4),
        d0 + DA(5),
    ])
    start_param = [x0, x0_slope, y0, y0_slope, d0]

    result = euler_dz(
        z0=z0,
        param=param,
        start_param=start_param,
        plane_dict=plane_dict,
        z_range=z_range,
        potential_func=potential_func,
        dz=dz,
        steps=steps,
        show_progress=True,
    )

    print("Final DA state:")
    print(result)


if __name__ == "__main__":
    main()
