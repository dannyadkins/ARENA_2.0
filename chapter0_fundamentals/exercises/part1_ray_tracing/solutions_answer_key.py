# %%

import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
# assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part1_ray_tracing', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"

# %%

def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    num_points = 2 # first one is origin, second one is a ray with x=1 and then each y from -y_limit to y_limit and z=0
    num_dim = 3
    rays = t.zeros(num_pixels, num_points, num_dim)
    ray_y_points = t.linspace(-1 * y_limit, y_limit, num_pixels)
    rays[:, 1, 1] = ray_y_points
    rays[:, 1, 0] = 1
    return rays

rays1d = make_rays_1d(9, 10.0)

# if MAIN:
#     fig = render_lines_with_plotly(rays1d)

# %%

if MAIN:
	fig = setup_widget_fig_ray()
	display(fig)
	
	@interact
	def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
		t.manual_seed(seed)
		L_1, L_2 = t.rand(2, 2)
		P = lambda v: L_1 + v * (L_2 - L_1)
		x, y = zip(P(-2), P(2))
		with fig.batch_update(): 
			fig.data[0].update({"x": x, "y": y}) 
			fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
			fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})

# %%

segments = t.tensor([
	[[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
	[[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
	[[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

# %%

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Using torch.lingalg.solve and torch.stack, implement the intersect_ray_1d function to solve the above matrix equation

    Return True if the ray intersects the segment.
    '''
    print("Ray: ", ray)
    print("Segment: ", segment)
    l1 = segment[0]
    l2 = segment[1]

    o = ray[0]
    d = ray[1]

    

    return True 

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)