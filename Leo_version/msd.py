import utils as ut

def msd_prep(inputdict):

    inputdict = ut.questions_dynamic_sim(inputdict)

    return inputdict

def msd_calc(inputdict):
    
        # Load the trajectory
        trajectory = ut.load_trajectory(inputdict['trajectory_file'])
    
        # Calculate the center of mass for each molecule
        mol_com = ut.COM_calculation(trajectory)
    
        # Calculate the minimum image distance
        distances = ut.minimum_image_distance(inputdict['box_dimension'], mol_com, mol_com)
    
        # Calculate the mean square displacement
        msd = ut.mean_square_displacement(distances, inputdict['dt'])
    
        return msd