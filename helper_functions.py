import numpy as np
from scipy import interpolate

from copy import deepcopy
import time
from tqdm import tqdm

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.csg import *

from netgen.occ import *
from netgen.meshing import IdentificationType



def get_idos_barycentric(pot_diff, pot_avg, pot_min, pot_max, en_min, en_max, idos_interpolated):


    restrict_domain(pot_diff, pot_min, pot_max)
    restrict_domain(pot_avg, en_min, en_max)

    
    return idos_interpolated(pot_diff, pot_avg)


def restrict_domain(array, min_val, max_val):

    ### numpy fancy idexing

    #https://stackoverflow.com/questions/19666626/replace-all-elements-of-numpy-array-that-are-greater-than-some-value

    array[array < min_val] = min_val
    array[array > max_val] = max_val

    return



### input: potential (gfu), and dampening factor alpha only relevant for first iteration if set
### fem_n: number of points on chess board, in 1-direction
### meshptsTop and meshptsBot: chessboard coordinates, top and bottom, defined with model later
### average_pot = 0.5 * ( \phi_top +\phi_bot), relevant for evaluating iDOS, or the charge density
### diff_pot = |\phi_top - \phi_bot|, changes bandgap and iDOS, only necessary for BLG graphene

def gen_charge(gfu, fem_n, meshptsTop, meshptsBot, pot_min, pot_max, en_min, en_max, idos_interpolated, alpha = 1):
    
    potTop = np.zeros((fem_n,fem_n), dtype=np.float64)
    potBottom = np.zeros((fem_n,fem_n), dtype=np.float64)
    
    potTop    = gfu(meshptsTop).reshape((fem_n,fem_n))
    potBottom = gfu(meshptsBot).reshape((fem_n,fem_n))
    
    average_pot  =  0.5*(potTop+potBottom) #gfu(meshptsMid).reshape((fem_n,fem_n))
    diff_pot = np.abs(potTop-potBottom)

    
    charge = get_idos_barycentric(diff_pot, alpha * average_pot, pot_min, pot_max, en_min, en_max, idos_interpolated)
    
    return charge, average_pot, diff_pot



### interpolation of charge density onto NGSolve gridfunctions
### charge is 2d array

def interpolate_charge(charge, gfQ, hBn_thickness, edge_length, blg_thickness):

    shape_0 = charge.shape[0]
    shape_1 = charge.shape[1]

    interpolated_charge = VoxelCoefficient((0, 0, hBn_thickness), 
                    (edge_length, edge_length, hBn_thickness + blg_thickness), charge.reshape((1, shape_1, shape_0)) ) #((1, fem_n, fem_n))

                                           #### VoxelCoeffient expects a 3d array, reshape
                                           
    gfQ.Set(interpolated_charge)




### input NGSolve gridfunction of potential gfu, output: 2 arrays, for average_pot and diff_pot

def gen_supercell_pot(gfu, nx, ny, supercell_Top, supercell_Bot):
    
    potTop    = np.zeros((ny, nx), dtype=np.float64)
    potBottom    = np.zeros((ny, nx), dtype=np.float64)
    
    potTop    = gfu(supercell_Top).reshape((ny, nx))
    potBottom = gfu(supercell_Bot).reshape((ny, nx))
    
    average_pot  =  0.5*(potTop+potBottom) #gfu(supercell_Mid).reshape((ny, nx))
    diff_pot = np.abs(potTop-potBottom)
    
    return average_pot, diff_pot



### given lattice constant, return BLG area and volume

def blg_area_vol (a_lattice, blg_grapheneheight, Angstrom):
    R1 = np.array([a_lattice, 0])

    rot_angle = (120.0 / 360.0) * 2.0 * np. pi

    rotmat = np.array([ [ np.cos(rot_angle), -np.sin(rot_angle)] , [ np.sin(rot_angle), np.cos(rot_angle)] ] )

    R2 = np.dot(rotmat, R1)

    R_matrix = np.array([R1,R2]).T

    G_matrix = np.linalg.inv(R_matrix)*2.0*np.pi

    G1 = G_matrix[0]
    G2 = G_matrix[1]


    area_matrix = np.zeros((3,3), dtype=np.float64)

    area_matrix[0][0] = R1[0]
    area_matrix[0][1] = R1[1]

    area_matrix[1][0] = R2[0]
    area_matrix[1][1] = R2[1]

    area_matrix[2][2] = 1


    volume_matrix = np.copy(area_matrix)

    volume_matrix[2][2] = blg_grapheneheight

    bilayer_graphene_area = np.abs(np.linalg.det(area_matrix)) *  pow(Angstrom,2) ### Angstrom^2
    bilayer_graphene_volume = np.abs(np.linalg.det(volume_matrix)) *  pow(Angstrom,3) ### Angstrom^3

    return bilayer_graphene_area, bilayer_graphene_volume