"""
Adapted from RGBD to PC code from http://www.open3d.org/docs/0.7.0/tutorial/Basic/rgbd_images/nyu.html
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import re
import matplotlib.image as mpimg

def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out

print("Read NYU dataset")
# Open3D does not support ppm/pgm file yet. Not using o3d.io.read_image here.
# MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.

color_raw = mpimg.imread("C:/Users/spathak/Downloads/GP_based_PCS_Final/improvements/nyu dataset/bedrooms_part1/bedroom_0007/r-1294886469.459506-961777627.ppm")
depth_raw = read_nyu_pgm("C:/Users/spathak/Downloads/GP_based_PCS_Final/improvements/nyu dataset/bedrooms_part1/bedroom_0007/d-1294886469.476187-962219264.pgm")
color = o3d.geometry.Image(color_raw)
depth = o3d.geometry.Image(depth_raw)
rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(color, depth)
print(rgbd_image)
plt.subplot(1, 2, 1)
plt.title('NYU grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('NYU depth image')
plt.imshow(rgbd_image.depth)
plt.show()
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd.paint_uniform_color(np.ndarray([3, 1]))
o3d.visualization.draw_geometries([pcd])

print(np.asarray(pcd.points).shape)
print(pcd.has_colors())
# o3d.io.write_point_cloud("C:/Users/spathak/Downloads/GP_based_PCS_Final/improvements/nyu_color_3.ply", pcd)