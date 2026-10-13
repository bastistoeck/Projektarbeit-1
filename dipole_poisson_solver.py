import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

import matplotlib as mpl

from scipy import interpolate
import scipy.optimize

from copy import deepcopy
import time
from tqdm import tqdm

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.csg import *

from netgen.occ import *
from netgen.meshing import IdentificationType



class dipole_poisson_solver:
    
    def __init__(self, dipole_moment_debye, dipole_d, dipole_pos, inplane_hBn_bulk, outofplane_hBn_bulk, inplane_graphene, outofplane_graphene, epsilon_0):
        # units from debye to our units
        self.dipole_moment = 0.20819427 * dipole_moment_debye
        self.dipole_d = dipole_d
        self.dipole_pos = dipole_pos
        # calculate dipole charge
        self.dipole_q = self.dipole_moment/self.dipole_d

        # dielectric constant and nanometers
        self.epsilon_0 = epsilon_0
        self.nm = 10

        # only non zero boundary condition
        self.back_gate_V = None
        # all other boundaries
        self.bc = 0.0

        # geometry parameters
        self.vac_box_width = None
        self.height_bottom = None
        self.height_top = None
        self.blg_height = None
        self.sphere_radius = None
        self.max_h = None
        self.V_sphere = None
        self.dipole_fes_order = None
        self.hbn_maxh_factor = None
        self.blg_maxh_factor = None

        # ngsolve objects
        self.mesh_dipole = None
        self.gfQ_dipole = None
        self.gfu = None
        self.gfud = None
        self.gfu_solveonce = None
        self.gfQ = None
        self.ndofs_potential_fes = None

        self.a = None
        self.inv = None

        self.fes = None
        self.fes_dipole = None
        self.fesQ = None

        # BLG parameters
        self.fem_n = None
        self.epsilon_z = None
        self.alpha = None
        self.idos_interpolated = None

        # solve once first
        self.solve_without_BLG_was_run = False

        # chessboard
        self.meshptsTop = None
        self.meshptsMid = None
        self.meshptsBot = None

        # dielectric tensor
        self.eps_dict = {"hBn" : CoefficientFunction((inplane_hBn_bulk, 0, 0,
                                        0, inplane_hBn_bulk, 0,
                                        0, 0, outofplane_hBn_bulk),dims=(3,3)),
                                    "BLG" : CoefficientFunction((inplane_graphene, 0, 0,
                                        0, inplane_graphene, 0,
                                        0, 0, outofplane_graphene),dims=(3,3)),
                                    "vac" : CoefficientFunction((1.0, 0, 0,
                                        0, 1.0, 0,
                                        0, 0, 1.0), dims=(3,3)),    
                                    "positive dipole" : CoefficientFunction((1.0, 0, 0,
                                        0, 1.0, 0,
                                        0, 0, 1.0), dims=(3,3)),
                                    "negative dipole" : CoefficientFunction((1.0, 0, 0,
                                        0, 1.0, 0,
                                        0, 0, 1.0), dims=(3,3))
                                    }
        self.epsr = None

        # solver objects
        self.global_it = [0]
        self.history_charges = None
        self.history_pots_av = None
        self.history_diff_pot = None
        self.a_vec = None
        self.increment = None

        self.solver_time = None
        self.total_charge = None
        self.total_charge_squared = None
        self.sol = None

        self.pot_min = None
        self.pot_max = None
        self.en_min = None
        self.en_max = None

        # induced dipole
        self.induced_charge = None
        self.sep_index = None

        self.Q_top = None
        self.p_top = None
        self.center_of_charge_top = None

        self.Q_bot = None
        self.p_bot = None
        self.center_of_charge_bot = None

        self.d_vec = None
        self.p_total = None
        self.separation = None


    def build_geometry(self, vac_box_width, height_bottom, height_top, blg_height, sphere_radius, max_h, hbn_maxh, blg_maxh, dipole_fes_order=3):
        """
        create ngsolve geometry with dipole as grid function; name boundaries
        """
        self.vac_box_width = vac_box_width
        self.height_bottom = height_bottom
        self.height_top = height_top
        self.blg_height = blg_height
        self.sphere_radius = sphere_radius
        self.max_h = max_h
        self.V_sphere = 4/3 * np.pi * sphere_radius ** 3
        self.dipole_fes_order = dipole_fes_order
        self.blg_maxh_factor = blg_maxh
        self.hbn_maxh_factor = hbn_maxh

        # bottom boxes
        box_width = self.vac_box_width
        hBn_bottom = Box(Pnt(0 ,0 ,0 ), Pnt(box_width, box_width, self.height_bottom))
        BLG = Box(Pnt(0 ,0 , self.height_bottom), Pnt(box_width, box_width, self.height_bottom + self.blg_height))
        hBn_top = Box(Pnt(0 ,0 , self.height_bottom + self.blg_height), Pnt(box_width, box_width, self.height_bottom + self.blg_height + self.height_top))

        # calculate dipole positon
        vac_origin = np.array([0.0, 0.0, self.height_bottom + self.blg_height + self.height_top])
        center_vec = vac_origin + np.array([0.5 * self.vac_box_width, 0.5 * self.vac_box_width, self.dipole_pos])
        sphere_center_vec = np.array([0.0, 0.5 * self.dipole_d, 0.0])
        pos_sphere_center = center_vec + sphere_center_vec
        neg_sphere_center = center_vec - sphere_center_vec

        # dipole in vacuum
        vac_box = Box(Pnt(vac_origin[0], vac_origin[1], vac_origin[2]), Pnt(self.vac_box_width, self.vac_box_width, self.vac_box_width + vac_origin[2]))
        pos_sphere = Sphere(Pnt(pos_sphere_center[0], pos_sphere_center[1], pos_sphere_center[2]), self.sphere_radius)
        neg_sphere = Sphere(Pnt(neg_sphere_center[0], neg_sphere_center[1], neg_sphere_center[2]), self.sphere_radius)

        # subtract spheres from vacuum
        vac_box = vac_box - pos_sphere - neg_sphere

        # materials
        hBn_bottom.mat("hBn")
        BLG.mat("BLG")
        hBn_top.mat("hBn")
        vac_box.mat("vac")
        pos_sphere.mat("positive dipole")
        neg_sphere.mat("negative dipole")

        # boundaries names
        hBn_bottom.faces.Min(Z).name = "bottom"
        vac_box.faces.Max(Z).name = "top"

        hBn_bottom.faces.Min(X).name = "bot_min_x"
        hBn_bottom.faces.Max(X).name = "bot_max_x"
        hBn_bottom.faces.Min(Y).name = "bot_min_y"
        hBn_bottom.faces.Max(Y).name = "bot_max_y"

        BLG.faces.Min(X).name = "blg_min_x"
        BLG.faces.Max(X).name = "blg_max_x"
        BLG.faces.Min(Y).name = "blg_min_y"
        BLG.faces.Max(Y).name = "blg_max_y"

        hBn_top.faces.Min(X).name = "top_min_x"
        hBn_top.faces.Max(X).name = "top_max_x"
        hBn_top.faces.Min(Y).name = "top_min_y"
        hBn_top.faces.Max(Y).name = "top_max_y"

        vac_box.faces.Min(X).name = "vac_min_x"
        vac_box.faces.Max(X).name = "vac_max_x"
        vac_box.faces.Min(Y).name = "vac_min_y"
        vac_box.faces.Max(Y).name = "vac_max_y"

        # generate geometry

        # finer maxh for blg and hbn (object.faces.maxh vs object.maxh?)
        BLG.maxh = self.blg_maxh_factor * self.max_h
        hBn_bottom.maxh = self.hbn_maxh_factor * self.max_h
        hBn_top.maxh = self.hbn_maxh_factor * self.max_h

        geo_dipole = OCCGeometry([vac_box, hBn_bottom, hBn_top, BLG, pos_sphere, neg_sphere])
        self.mesh_dipole = Mesh(geo_dipole.GenerateMesh(maxh=self.max_h)).Curve(3)

        # dipole grid function
        self.fes_dipole = L2(self.mesh_dipole, order=self.dipole_fes_order)
        dipole_charges = {
            "positive dipole": self.dipole_q/self.V_sphere,
            "negative dipole": -1 * self.dipole_q/self.V_sphere,
            "vac": 0.0,
            "hBn": 0.0,
            "BLG": 0.0
        }

        cf_dipole = self.mesh_dipole.MaterialCF(dipole_charges, default=0)
        self.gfQ_dipole = GridFunction(self.fes_dipole)
        self.gfQ_dipole.Set(cf_dipole)

        # set epsilon tensor
        self.epsr = self.mesh_dipole.MaterialCF( self.eps_dict )

    
    def solve_without_BLG (self, back_gate_V, gfu_fes_order=2):
        """
        solve with dipole only; no charges on BLG
        
        solution is stored as self.gfu_solveonce

        calculates the inverse of matrix a
        """
        # boundary dictionary
        self.back_gate_V = back_gate_V
        bound_dict = {'bottom' : self.back_gate_V, 'top' : self.bc,
                            'top_max_x' : self.bc, 'top_min_x' : self.bc, 'top_max_y' : self.bc, 'top_min_y' : self.bc,
                            'bot_max_x' : self.bc, 'bot_min_x' : self.bc, 'bot_max_y' : self.bc, 'bot_min_y' : self.bc,
                            'blg_max_x' : self.bc, 'blg_min_x' : self.bc, 'blg_max_y' : self.bc, 'blg_min_y' : self.bc,
                            'vac_max_x' : self.bc, 'vac_min_x' : self.bc, 'vac_max_y' : self.bc, 'vac_min_y' : self.bc
                        }
        # finite element space for potential
        self.fes = H1(self.mesh_dipole, order=gfu_fes_order, dirichlet="bottom|top|top_max_x|top_min_x|top_max_y|top_min_y|bot_max_x|bot_min_x|bot_max_y|bot_min_y|blg_max_x|blg_min_x|blg_max_y|blg_min_y|vac_max_x|vac_min_x|vac_max_y|vac_min_y")
        self.ndofs_potential_fes = self.fes.ndof
        u, v = self.fes.TnT()
        self.a = BilinearForm(self.epsilon_0 * self.epsr * grad(u) * grad(v) * dx).Assemble()
        f = LinearForm(self.gfQ_dipole * v * dx("positive dipole|negative dipole")).Assemble()

        self.gfu_solveonce = GridFunction(self.fes)
        self.gfud = self.mesh_dipole.BoundaryCF(bound_dict, default=0)
        self.gfu_solveonce.Set(self.gfud, BND)

        self.inv = self.a.mat.Inverse(freedofs=self.fes.FreeDofs(), inverse="sparsecholesky")

        res = (self.a.mat * self.gfu_solveonce.vec - f.vec).Evaluate()

        self.gfu_solveonce.vec.data -= self.inv * res

        # save solution:
        #self.gfu_solveonce = GridFunction(self.fes)
        #self.gfu_solveonce.Set(self.gfu)
        self.global_it[0] = 0
        self.solve_without_BLG_was_run = True


    def solve(self, fem_n, epsilon_z, idos_interpolated, pot_min, pot_max, en_min, en_max, use_abs=False ,starting_alpha=0.01):
        """
        solve non-linear poisson equation on given geometry;
        idos_interpolated is the integrated density of states of BLG
        
        """
        # check if solve once was run before
        if self.solve_without_BLG_was_run == False:
            raise Exception('solve_without_BLG() needs to be run before running solve()')
        
        self.fem_n = fem_n
        self.epsilon_z = epsilon_z
        self.alpha = starting_alpha
        self.idos_interpolated = idos_interpolated

        self.pot_min = pot_min
        self.pot_max = pot_max
        self.en_min = en_min
        self.en_max = en_max

        # create chessboard for charges on BLG
        edge_length = self.vac_box_width
        hBn_thickness = self.height_bottom
        blg_thickness = self.blg_height

        self.increment = edge_length / (self.fem_n - 1)

        xs   = np.linspace(0, edge_length, self.fem_n)
        ys   = np.linspace(0, edge_length, self.fem_n)

        zs_Bot = np.array(hBn_thickness + self.epsilon_z)

        zs_Mid = np.array(hBn_thickness + blg_thickness/2)

        zs_Top = np.array(hBn_thickness + blg_thickness - self.epsilon_z)

        dummy_m_x, dummy_m_y, dummy_m_z_b = np.meshgrid(xs, ys, zs_Bot)

        dummy_m_z_m = np.ones((fem_n,fem_n,1), dtype=np.float64) * zs_Mid
        dummy_m_z_t = np.ones((fem_n,fem_n,1), dtype=np.float64) * zs_Top

        ptsBot = np.array([np.ravel(dummy_m_x), np.ravel(dummy_m_y), np.ravel(dummy_m_z_b)]).T
    
        ptsMid = np.array([np.ravel(dummy_m_x), np.ravel(dummy_m_y), np.ravel(dummy_m_z_m)]).T

        ptsTop = np.array([np.ravel(dummy_m_x), np.ravel(dummy_m_y), np.ravel(dummy_m_z_t)]).T
            
        self.meshptsTop = self.mesh_dipole(ptsTop[:,0], ptsTop[:,1], ptsTop[:,2])
        self.meshptsMid = self.mesh_dipole(ptsMid[:,0], ptsMid[:,1], ptsMid[:,2])
        self.meshptsBot = self.mesh_dipole(ptsBot[:,0], ptsBot[:,1], ptsBot[:,2])

        # generate first charges on BLG
        self.fesQ = L2(self.mesh_dipole, order=3, definedon="BLG")
        self.gfQ = GridFunction(self.fesQ)

        self.gfu = GridFunction(self.fes)
        self.gfu.Set(self.gfu_solveonce)

        charge, average_pot, diff_pot = self.gen_charge(self.alpha) #0.01 #0.00001 #0.00001; alphas 

        self.history_charges = [charge]
        self.history_pots_av = [average_pot]
        self.history_diff_pot = [diff_pot]

        self.a_vec = [self.a]

        charge = np.ravel(charge)

        start_time = time.time()

        self.sol = scipy.optimize.root(self.residual_of_charge, 
                        charge,  args=(False),
                        method='df-sane', options = {"fatol": pow(10,-9)})  # -9
        
        end_time = time.time()
        self.solver_time = end_time - start_time

        self.total_charge = Integrate(self.gfQ, self.mesh_dipole, definedon=self.mesh_dipole.Materials("BLG"))
        self.total_charge_squared = Integrate(self.gfQ * self.gfQ, self.mesh_dipole, definedon=self.mesh_dipole.Materials("BLG"))

        # calculate induced dipole moment in BLG
        if use_abs == False:
            self.induced_dipole()
        elif use_abs == True:
            self.induced_dipole_abs()

        #print("solver takes ", self.solver_time, " seconds and ", self.global_it[0], " iterations")

        #print("charge =", self.total_charge)


    # function to minimize
    def residual_of_charge (self, charge, show_plots=False):
        # a = self.a_vec[0] # why would I need that?

        charge = charge.reshape((self.fem_n, self.fem_n))

        u, v = self.fes.TnT()

        self.interpolate_charge(charge)

        #update right side of poisson equation
        with TaskManager():
            
            #f   = LinearForm( gfQ * v * dx("bilayer_graphene")).Assemble() # old version for f
            # new f with gfQ<_dipole: 
            f = LinearForm(self.fes)
            f += self.gfQ_dipole * v * dx("positive dipole|negative dipole")
            f += self.gfQ * v * dx("BLG")
            f.Assemble()

            self.gfu.Set(self.gfud, BND)
        

            res = (self.a.mat * self.gfu.vec - f.vec).Evaluate()
            self.gfu.vec.data -= self.inv * res

        new_charge, new_average_pot, new_diff_pot = self.gen_charge()

        # count iterations
        n_it = self.global_it[0]
        self.global_it[0] += 1

        if show_plots == True:
    
            #plot figures
            plt.figure(figsize=(11,8))
            gs=gridspec.GridSpec(1,4, wspace = 0.6) 
            ax = plt.subplot(gs[0,0])
            img = ax.imshow(charge*pow(self.nm,3), origin="lower")
            plt.colorbar(img, ax=ax,fraction=0.046, pad=0.04)
            ax2 = plt.subplot(gs[0,1])
            img2 = ax2.imshow(new_charge*pow(self.nm,3), origin="lower")
            plt.colorbar(img2, ax=ax2,fraction=0.046, pad=0.04)
            ax3 = plt.subplot(gs[0,2])
            img3 = ax3.imshow((charge - new_charge) * pow(self.nm,3), origin="lower")
            plt.colorbar(img3, ax=ax3,fraction=0.046, pad=0.04)
            ax4 = plt.subplot(gs[0,3])
            img4 = ax4.imshow(new_average_pot, origin="lower")
            plt.colorbar(img4, ax=ax4,fraction=0.046, pad=0.04)
            plt.suptitle("iteration " + str(n_it) + "  "+  r" $\rho_{old}$" + " , " +\
                    r" $\rho_{new}$" + " , " + r" $\Delta\rho$ and" +\
                    r" $\phi_{new}$", y=0.7)
            plt.show()
        
        self.history_charges[0] = [new_charge]
        self.history_pots_av[0] = [new_average_pot]
        self.history_diff_pot[0] = [new_diff_pot]

        integrated_charge = self.blg_height*np.trapezoid(np.trapezoid(charge, dx=self.increment), dx=self.increment)
        integrated_new_charge = self.blg_height*np.trapezoid(np.trapezoid(new_charge, dx=self.increment), dx=self.increment)

        charge = np.ravel(charge)

        new_charge = np.ravel(new_charge)

        #print("iteration: ", n_it)
        #print("#", n_it, "  rho_n-1 - rho_n (max): ", np.max(abs(charge - new_charge)))
        #print("#", n_it, "  |q_n-1 - q_n| : ", abs(integrated_charge - integrated_new_charge))
        return  (charge - new_charge)


    # gen charge plus dependecies
    def restrict_domain(self, array, min_val, max_val):
        ### numpy fancy idexing

        #https://stackoverflow.com/questions/19666626/replace-all-elements-of-numpy-array-that-are-greater-than-some-value

        array[array < min_val] = min_val
        array[array > max_val] = max_val

        return

    def get_idos_barycentric(self, pot_diff, pot_avg):


        self.restrict_domain(pot_diff, self.pot_min, self.pot_max)
        self.restrict_domain(pot_avg, self.en_min, self.en_max)

        
        return self.idos_interpolated(pot_diff, pot_avg)

    def gen_charge(self, alpha_gen_charge=1):
        potTop = np.zeros((self.fem_n, self.fem_n), dtype=np.float64)
        potBottom = np.zeros((self.fem_n, self.fem_n), dtype=np.float64)
        
        potTop    = self.gfu(self.meshptsTop).reshape((self.fem_n, self.fem_n))
        potBottom = self.gfu(self.meshptsBot).reshape((self.fem_n, self.fem_n))
        
        average_pot  =  0.5*(potTop+potBottom) #gfu(meshptsMid).reshape((fem_n,fem_n))
        diff_pot = np.abs(potTop-potBottom)

        
        charge = self.get_idos_barycentric(diff_pot, alpha_gen_charge * average_pot)
        
        return charge, average_pot, diff_pot
    
    def interpolate_charge(self, charge):
        shape_0 = charge.shape[0]
        shape_1 = charge.shape[1]

        interpolated_charge = VoxelCoefficient((0, 0, self.height_bottom), 
                        (self.vac_box_width, self.vac_box_width, self.height_bottom + self.blg_height), charge.reshape((1, shape_1, shape_0)) ) #((1, fem_n, fem_n))

                                            #### VoxelCoeffient expects a 3d array, reshape
                                            
        self.gfQ.Set(interpolated_charge)

    def induced_dipole(self):
        self.induced_charge = self.gfQ(self.meshptsMid).reshape((self.fem_n, self.fem_n))
        middle_index = math.floor(self.induced_charge.shape[0]/2)
        y_cut = self.induced_charge[:,middle_index]

        # cut boundaries; finite size effects; 10% is removed
        cut_index = int(0.05 * self.induced_charge.shape[0])
        self.induced_charge = self.induced_charge[cut_index:self.fem_n - cut_index, cut_index:self.fem_n - cut_index]
    
        # cut more here, in order to make it easier to find the inflection point
        larger_cut_percentage = 0.3
        larger_cut_index = int(larger_cut_percentage/2 * self.fem_n)
        y_cut = y_cut[larger_cut_index:self.fem_n - larger_cut_index]

        # x values
        dx = self.vac_box_width/(self.fem_n - 1)
        x_max = (self.fem_n - 1) * dx
        x = np.linspace(0, x_max, self.fem_n)
        x = x[larger_cut_index:self.fem_n - larger_cut_index]

        # regression to get smooth derivative
        spline = interpolate.make_smoothing_spline(x, y_cut, lam=100)
        x_new = np.linspace(x[0], x[-1], 2000)

        # second derivative
        y_double_prime = spline(x_new, nu=2)

        max_index = np.argmax(y_double_prime)
        min_index = np.argmin(y_double_prime)

        if max_index < min_index:
            sep_index_temp = max_index + np.argmin(abs(y_double_prime[max_index:min_index]))
        elif min_index < max_index:
            sep_index_temp = min_index + np.argmin(abs(y_double_prime[min_index:max_index]))
        
        sep_index_temp = np.argmin(abs(x - x_new[sep_index_temp]))
        
        self.sep_index = larger_cut_index - cut_index + sep_index_temp

        # split charge array
        top_charge = self.induced_charge[:self.sep_index, :]
        bot_charge = self.induced_charge[self.sep_index:, :]

        ### find centers of charges

        dx = self.vac_box_width/(self.fem_n - 1)
        x_max = (self.fem_n - 1) * dx
        x = np.linspace(0, x_max, self.fem_n)
        y = np.linspace(0, x_max, self.fem_n)

        # cut x and y; ignore boundary effects
        x = x[cut_index:self.fem_n - cut_index]
        y = y[cut_index:self.fem_n - cut_index]

        # meshgrids
        x_top = x[:self.sep_index]
        x_bot = x[self.sep_index:]
        X_top, Y_top = np.meshgrid(x_top, y, indexing='ij')
        X_bot, Y_bot = np.meshgrid(x_bot, y, indexing='ij')

        # top
        p_top_x = np.trapezoid(np.trapezoid(top_charge * X_top, dx=dx, axis=0), dx=dx, axis=0)
        p_top_y = np.trapezoid(np.trapezoid(top_charge * Y_top, dx=dx, axis=0), dx=dx, axis=0)

        self.Q_top = np.trapezoid(np.trapezoid(top_charge, dx=dx, axis=0), dx=dx, axis=0) * self.blg_height

        self.p_top = (np.array([p_top_x, p_top_y]) * self.blg_height)
        self.center_of_charge_top = self.p_top/self.Q_top

        # bottom
        p_bot_x = np.trapezoid(np.trapezoid(bot_charge * X_bot, dx=dx, axis=0), dx=dx, axis=0)
        p_bot_y = np.trapezoid(np.trapezoid(bot_charge * Y_bot, dx=dx, axis=0), dx=dx, axis=0)

        self.Q_bot = np.trapezoid(np.trapezoid(bot_charge, dx=dx, axis=0), dx=dx, axis=0) * self.blg_height

        self.p_bot = (np.array([p_bot_x, p_bot_y]) * self.blg_height)
        self.center_of_charge_bot = self.p_bot/self.Q_bot

        # induced dipole
        self.p_total = self.p_top + self.p_bot
        self.d_vec = self.center_of_charge_top - self.center_of_charge_bot
        self.separation = np.linalg.norm(self.d_vec)


    def induced_dipole_abs(self):
        self.induced_charge = self.gfQ(self.meshptsMid).reshape((self.fem_n, self.fem_n))
        middle_index = math.floor(self.induced_charge.shape[0]/2)
        y_cut = self.induced_charge[:,middle_index]
        
        # cut boundaries; finite size effects; 10% is removed
        cut_index = int(0.05 * self.induced_charge.shape[0])
        self.induced_charge = self.induced_charge[cut_index:self.fem_n - cut_index, cut_index:self.fem_n - cut_index]
            
        # cut more here, in order to make it easier to find the inflection point
        larger_cut_percentage = 0.3
        larger_cut_index = int(larger_cut_percentage/2 * self.fem_n)
        y_cut = y_cut[larger_cut_index:self.fem_n - larger_cut_index]
        
        # x values
        dx = self.vac_box_width/(self.fem_n - 1)
        x_max = (self.fem_n - 1) * dx
        x = np.linspace(0, x_max, self.fem_n)
        x = x[larger_cut_index:self.fem_n - larger_cut_index]
        
        # regression to get smooth derivative
        spline = interpolate.make_smoothing_spline(x, y_cut, lam=100)
        x_new = np.linspace(x[0], x[-1], 2000)
        
        # second derivative
        y_double_prime = spline(x_new, nu=2)
        
        max_index = np.argmax(y_double_prime)
        min_index = np.argmin(y_double_prime)
        
        if max_index < min_index:
            sep_index_temp = max_index + np.argmin(abs(y_double_prime[max_index:min_index]))
        elif min_index < max_index:
            sep_index_temp = min_index + np.argmin(abs(y_double_prime[min_index:max_index]))
                
        sep_index_temp = np.argmin(abs(x - x_new[sep_index_temp]))
                
        self.sep_index = larger_cut_index - cut_index + sep_index_temp
        
        # split charge array
        top_charge = self.induced_charge[:self.sep_index, :]
        bot_charge = self.induced_charge[self.sep_index:, :]
        
        ### find centers of charges
        
        dx = self.vac_box_width/(self.fem_n - 1)
        x_max = (self.fem_n - 1) * dx
        x = np.linspace(0, x_max, self.fem_n)
        y = np.linspace(0, x_max, self.fem_n)
        
        # cut x and y; ignore boundary effects
        x = x[cut_index:self.fem_n - cut_index]
        y = y[cut_index:self.fem_n - cut_index]
        
        # meshgrids
        x_top = x[:self.sep_index]
        x_bot = x[self.sep_index:]
        X_top, Y_top = np.meshgrid(x_top, y, indexing='ij')
        X_bot, Y_bot = np.meshgrid(x_bot, y, indexing='ij')
        
        # top
        p_top_x = np.trapezoid(np.trapezoid(abs(top_charge) * X_top, dx=dx, axis=0), dx=dx, axis=0)
        p_top_y = np.trapezoid(np.trapezoid(abs(top_charge) * Y_top, dx=dx, axis=0), dx=dx, axis=0)
        
        self.Q_top = np.trapezoid(np.trapezoid(abs(top_charge), dx=dx, axis=0), dx=dx, axis=0) * self.blg_height
        
        self.p_top = (np.array([p_top_x, p_top_y]) * self.blg_height)
        self.center_of_charge_top = self.p_top/self.Q_top
        
        # bottom
        p_bot_x = np.trapezoid(np.trapezoid(abs(bot_charge) * X_bot, dx=dx, axis=0), dx=dx, axis=0)
        p_bot_y = np.trapezoid(np.trapezoid(abs(bot_charge) * Y_bot, dx=dx, axis=0), dx=dx, axis=0)
        
        self.Q_bot = np.trapezoid(np.trapezoid(abs(bot_charge), dx=dx, axis=0), dx=dx, axis=0) * self.blg_height
        
        self.p_bot = (np.array([p_bot_x, p_bot_y]) * self.blg_height)
        self.center_of_charge_bot = self.p_bot/self.Q_bot
        
        # induced dipole
        self.p_total = self.p_top + self.p_bot
        self.d_vec = self.center_of_charge_top - self.center_of_charge_bot
        self.separation = np.linalg.norm(self.d_vec)
        

