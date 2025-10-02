# install cgal using conda:  conda install -c conda-forge cgal

from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Point_set_processing_3 import *
import time


# wlop gives simp very similar to random_simp, is good for noisy datasets
# dragon_vrip size = 1000, maximum_variation = 0.01
# armadillo size = 1000, maximum_variation = 0.008
# lucy = size = 1000, maximum_variation = 0.05
# bunny = size = 100, maximum_variation = 0.001
# noise white 1e-02 armadillo = size = 1000, maximum_variation = 0.042
# 36 0.2 = size = 1000, maximum_variation = 0.0125
# 36 0.3 = size = 1000, maximum_variation = 0.002
# 36 0.4 = size = 100, maximum_variation = 0.00005
# 73 0.2 = size = 1000, maximum_variation = 0.0034
# 73 0.3 = size = 1000, maximum_variation = 0.0006
# 73 0.4 = size = 100, maximum_variation = 0.000018
# angel 0.04 = size = 1000, maximum_variation = 0.0097
# angel 0.1 = size = 1000, maximum_variation = 0.0039
# angel 0.06 = size = 1000, maximum_variation = 0.0069
# angel 0.05 = size = 1000, maximum_variation = 0.0082
# angel 0.03 = size = 1000, maximum_variation = 0.012
# angel 0.01 = size = 1000, maximum_variation = 0.0261

# seppe 0.06 size = 1000, maximum_variation = 0.0045
# bikeframe_high 0.3 size = 1000, maximum_variation = 0.00035
# nyu 0.04 size = 1000, maximum_variation = 0.029


t1= time.time()
print("Running read_xyz_points...")
points = Point_set_3("resources/clouds/nyu_color.ply")
print(points.size(), "points read")

print("Running wlop_simplify_and_regularize_point_set...")
wlop_point_set = Point_set_3()
wlop_simplify_and_regularize_point_set(points, wlop_point_set, select_percentage = 4, require_uniform_sampling = False)  # Output
print("Output of WLOP has", wlop_point_set.size(), "points")
t2= time.time()
print("Writing to ply")
wlop_point_set.write("resources/results/seppe_0.06_wlop.ply")
print(t2-t1)

# hierarchy doesnt give user specified simplified pc size, so one has to adjust size and max_variation to get desired size (below done to get approx 0.03 of drag_vrip)
# t3= time.time()
# print("Running read_xyz_points...")
# points = Point_set_3("resources/clouds/seppe.ply")
# print(points.size(), "points read")

# print("Running hierarchy_simplify_point_set...")
# hierarchy_simplify_point_set(points, size = 1000, maximum_variation = 0.0045)
# print(points.size(), "point(s) remaining,", points.garbage_size(),
#       "point(s) removed")
# t4= time.time()

# print(t4-t3)
# print("Writing to ply")
# points.write("resources/results/seppe_0.06_hc.ply")