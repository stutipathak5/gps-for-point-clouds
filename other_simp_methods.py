# install cgal using conda:  conda install -c conda-forge cgal

from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Point_set_processing_3 import *


# wlop gives simp very similar to random_simp, is good for noisy datasets

print("Running read_xyz_points...")
points = Point_set_3("resources/clouds/dragon_vrip.ply")
print(points.size(), "points read")

print("Running wlop_simplify_and_regularize_point_set...")
wlop_point_set = Point_set_3()
wlop_simplify_and_regularize_point_set(points, wlop_point_set, select_percentage = 3, require_uniform_sampling = False)  # Output
print("Output of WLOP has", wlop_point_set.size(), "points")

print("Writing to ply")
wlop_point_set.write("resources/results/dragon_vrip_wlop.ply")


# hierarchy doesnt give user specified simplified pc size, so one has to adjust size and max_variation to get desired size (below done to get approx 0.03 of drag_vrip)

print("Running read_xyz_points...")
points = Point_set_3("resources/clouds/dragon_vrip.ply")
print(points.size(), "points read")

print("Running hierarchy_simplify_point_set...")
hierarchy_simplify_point_set(points, size = 1000, maximum_variation = 0.01)
print(points.size(), "point(s) remaining,", points.garbage_size(),
      "point(s) removed")

print("Writing to ply")
points.write("resources/results/dragon_vrip_hier.ply")