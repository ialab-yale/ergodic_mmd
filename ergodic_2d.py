import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from solver_jaxopt import AugmentedLagrangeSolver

class EMMD():
    def __init__(self, input_args, x_0=None):
        num_points = input_args.get('num_points', 1000)
        self.points = np.random.uniform(-1, 1, size=(num_points, 2))
        
        self.T = input_args.get('T', 200)
        self.h = input_args.get('h', 0.001)
        self.max_dx = input_args.get('max_dx', 0.05)
        
        self.x_0 = np.array([-1.0, -1.0])
        self.x_f = np.array([1.0, 1.0])
        if x_0 is not None:
            self.x_0 = x_0
        
        self.X = jnp.linspace(self.x_0, self.x_f, num=self.T)
        
        self.info_dist = input_args.get('info_dist', lambda x: 1.0)
        self.P_XI = vmap(self.info_dist, in_axes=(0,))(self.points)
        self.P_XI = self.P_XI / jnp.sum(self.P_XI)  

    def RBF_kernel(self, x, xp, h=0.01):
        return jnp.exp(
            -jnp.sum((x - xp)**2) / h
        )

    def create_kernel_matrix(self, kernel_func, args=0):
        return vmap(vmap(kernel_func, in_axes=(0, None, None)), in_axes=(None, 0, None))
    
    def solve(self):
        args = {'h': self.h, 'points': self.points, 'P_XI': self.P_XI, 'T': self.T}
        self.params = {'X': self.X}

        KernelMatrix = self.create_kernel_matrix(self.RBF_kernel, args=args)
        
        # Define Loss and Constraints --------------------------------------------------------------------
        emmd_loss = lambda params, args: (
            np.sum(KernelMatrix(params['X'], params['X'], args['h'])) / (args['T']**2)
            - 2 * np.sum(args['P_XI'] @ KernelMatrix(params['X'], args['points'], args['h'])) / args['T']
            + 2 * jnp.mean(jnp.square(params['X'][1:] - params['X'][:-1]))
        )

        eq_constr = lambda params, args: jnp.array(0.)

        ineq_constr = lambda params, args: jnp.square(params['X'][1:] - params['X'][:-1]) - self.max_dx**2
        
        # ------------------------------------------------------------------------------------------------

        solver = AugmentedLagrangeSolver(self.params, emmd_loss, eq_constr, ineq_constr, args=args)

        solver.solve(max_iter=1000, eps=1e-8)

        self.trajectory = solver.solution['X']
        return self.trajectory

    def plot_2d(self):
        x = np.linspace(-1, 1, 400)
        y = np.linspace(-1, 1, 400)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        
        P_XI_grid = vmap(self.info_dist)(grid_points)
        P_XI_grid = P_XI_grid.reshape(X.shape)
        
        plt.figure(figsize=(10, 8))
        
        mesh = plt.pcolormesh(X, Y, P_XI_grid, shading='auto', cmap='gray', vmin=0, vmax=2)
        
        plt.contour(X, Y, P_XI_grid, levels=6, colors='grey', linewidths=0.8, linestyles='-')
        
        plt.colorbar(mesh, label='Utility')
        
        plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
        plt.scatter(self.trajectory[-1, 0], self.trajectory[-1, 1], color='r', marker='^', s=100, label='End')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
            
if __name__ == '__main__':
    # Define utility distribution
    centers = jnp.array([
        [0.3, 0.3],
        [0.0, -0.3],
        [0.0, 0.0],
        [-0.3, 0.3],
        [0.15, 0.15],
        [-0.15, 0.15],
        [0, -0.15]
    ])

    bw = 0.02

    # Define args
    args = {
        'num_points': 1000,
        'T': 100,
        'h': 0.03,
        'max_dx': 0.05,
        'info_dist': lambda x: jnp.sum(jnp.exp(-jnp.sum((x - centers)**2, axis=1) / bw))
    }

    emmd = EMMD(args)
    emmd.solve()
    emmd.plot_2d()