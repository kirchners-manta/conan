import os
import random

from conan.playground.doping_experiment import GrapheneSheet
from conan.playground.graph_utils import write_xyz


def create_graphene_sheets(num_sheets=100, output_folder="graphene_sheets"):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_sheets):
        # Create a graphene sheet with a size of 20x20 atoms
        graphene = GrapheneSheet(bond_distance=1.42, sheet_size=(20, 20))

        # Generate a random nitrogen doping percentage between 5% and 15%
        nitrogen_percentage = random.uniform(5, 15)

        # Add nitrogen doping to the graphene sheet
        graphene.add_nitrogen_doping(total_percentage=nitrogen_percentage, adjust_positions=False)

        # Save the graphene sheet as an XYZ file
        filename = os.path.join(output_folder, f"graphene_{i + 1}.xyz")
        write_xyz(graphene.graph, filename)

    print(
        f"\n{num_sheets} Graphene sheets with varying nitrogen doping have been created and saved in '{output_folder}'."
    )


if __name__ == "__main__":
    # Create 100 graphene sheets with varying nitrogen doping
    create_graphene_sheets(num_sheets=100, output_folder="graphene_sheets")
