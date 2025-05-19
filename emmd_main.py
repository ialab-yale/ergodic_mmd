import numpy as np
import jax.numpy as jnp
from jax import vmap
import trimesh
import open3d as o3d
import plotly.graph_objects as go

from solver_jaxopt import AugmentedLagrangeSolver

class EMMD():
    def __init__(self, input_args, x_0=None):
        self.load_mesh(input_args.get('mesh_path', 'obj_files/bunny.obj'))

        self.T = input_args.get('T', 200)
        self.h = input_args.get('h', 0.001)
        self.max_dx = input_args.get('max_dx', 0.1)
        self.push = input_args.get('push', 0.05)
        
        bbox = self.mesh.get_axis_aligned_bounding_box()
        bounds_min = np.array(bbox.min_bound)
        bounds_max = np.array(bbox.max_bound)

        if x_0 is None:
            self.x_0 = bounds_min
        else:
            self.x_0 = x_0

        self.x_f = bounds_max
        self.X = jnp.linspace(self.x_0, self.x_f, num=self.T)

        self.info_dist = input_args.get('info_dist', lambda x : 1)
        self.P_XI = vmap(self.info_dist, in_axes=(0,))(self.points)
        self.P_XI = self.P_XI / jnp.sum(self.P_XI)  

    def load_mesh(self, mesh_path="bunny.obj"):

        try:
            self.mesh = o3d.io.read_triangle_mesh(mesh_path)
            if len(pcd.points) == 0:
                raise ValueError("Point cloud is empty.")
        except Exception as e:
            print(f"Open3D load failed ({e}), trying trimesh")
            tri = trimesh.load(mesh_path)
            if isinstance(tri, trimesh.Scene):
                tri = trimesh.util.concatenate(tuple(tri.geometry.values()))
            self.mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.asarray(tri.vertices)),
                triangles=o3d.utility.Vector3iVector(np.asarray(tri.faces))
            )

        mesh_verts = np.asarray(self.mesh.vertices)
        centroid   = mesh_verts.mean(axis=0)
        mesh_verts = mesh_verts - centroid

        max_dim = np.max(mesh_verts.max(axis=0) - mesh_verts.min(axis=0))
        mesh_verts = mesh_verts / max_dim

        self.mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
        self.mesh.compute_vertex_normals()

        try:
            pcd = self.mesh.sample_points_uniformly(number_of_points=1000)
        except AttributeError:
            pcd = o3d.geometry.PointCloud(self.mesh.vertices)
            if len(pcd.points) > 1000:
                pcd = pcd.random_down_sample(1000/len(pcd.points))

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        pcd.normalize_normals()

        self.points  = np.asarray(pcd.points)
        self.normals = np.asarray(pcd.normals)
    
    def RBF_kernel(self, x, xp, h=0.01):
        return jnp.exp(
            -jnp.sum((x-xp)**2)/h
        )

    def create_kernel_matrix(self, kernel_func, args=0):
        return vmap(vmap(kernel_func, in_axes=(0, None, None)), in_axes=(None, 0, None))
    
    def solve(self):
        self.args = {'h' : self.h, 'points' : self.points + self.push*self.normals, 'P_XI' : self.P_XI, 'T' : self.T}
        self.params = {'X' : self.X}

        KernelMatrix = self.create_kernel_matrix(self.RBF_kernel, args=self.args)
        
        # Define Loss and Constraints --------------------------------------------------------------------
        emmd_loss = lambda params, args: (
            np.sum(KernelMatrix(params['X'], params['X'], args['h'])) / (args['T']**2)
            - 2 * np.sum(args['P_XI'] @ KernelMatrix(params['X'], args['points'], args['h'])) / args['T']
            + 2 * jnp.mean(jnp.square(params['X'][1:] - params['X'][:-1]))
        )

        eq_constr = lambda params, args: jnp.array(0.)

        ineq_constr = lambda params, args: jnp.square(params['X'][1:] - params['X'][:-1]) - self.max_dx**2

        # ------------------------------------------------------------------------------------------------

        solver = AugmentedLagrangeSolver(self.params, emmd_loss, eq_constr, ineq_constr, args = self.args)

        solver.solve(max_iter=1000,eps=1e-8)

        self.trajectory = solver.solution['X']

        return self.trajectory

    def plot_system(self, title="Ergodic MMD"):
        mesh = self.mesh
        trajectory = self.trajectory
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        P_XI = vmap(lambda x : 1, in_axes=(0,))(mesh.vertices) # Uniform
        P_XI = np.asarray(P_XI)  

        mesh_trace = go.Mesh3d(
            x=vertices[:, 0], 
            y=vertices[:, 1], 
            z=vertices[:, 2],
            i=faces[:, 0], 
            j=faces[:, 1], 
            k=faces[:, 2],
            intensity=P_XI,
            colorscale='reds', 
            intensitymode='vertex',
            opacity=0.6,
            flatshading=False,
            lighting=dict(ambient=0.3, diffuse=0.9, fresnel=0.1, roughness=0.3, specular=0.5),
            lightposition=dict(x=100, y=200, z=300),
            name="Mesh",
            showscale=True, 
            cmin=-1,
            cmax=1.5
        )

        trajectory_trace = go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode='lines',
            line=dict(color='#1f77b4', width=12),
            name="Trajectory"
        )

        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    title="",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor='white',
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False, 
                    range=[-1, 1],
                    showticklabels=False,
                    ticks=''
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor='white',
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,
                    range=[-1, 1],
                    showticklabels=False,
                    ticks=''
                ),
                zaxis=dict(
                    title="",
                    showgrid=False,
                    gridcolor='lightgray',
                    zerolinecolor='white',
                    showbackground=True,
                    backgroundcolor="white",
                    showspikes=False,
                    range=[-1, 1],
                    showticklabels=False,
                    ticks=''
                ),
                aspectmode='cube', 
            ),
            legend=dict(
                x=0.8, y=0.9, 
                bgcolor='rgba(255,255,255,0.8)',  
                bordercolor='black', 
                borderwidth=1
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(0,0,0,0)',  
        )

        fig = go.Figure(data=[mesh_trace, trajectory_trace], layout=layout)
        fig.show()
            
if __name__ == '__main__':
    mdl_path = 'obj_files/tortoise.obj'

    args = {'mesh_path': mdl_path, 
            'push': 0.2, 
            'h': 0.01,
            'T': 200, 
            'max_dx': 0.005}

    emmd = EMMD(args)
    emmd.solve()
    emmd.plot_system()
