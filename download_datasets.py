from torch_geometric.datasets import TOSCA
from torch_geometric.datasets import DynamicFAUST
from torch_geometric.datasets import ModelNet
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import torch_geometric

dataset=ModelNet(root="other_datasets/ModelNet40", name= '40', train= True)

# dataset = TOSCA(root="other_datasets/TOSCA")
# points = dataset[36].pos.numpy()
# print(points.shape[0])

# import numpy as np
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2], s=0.1)
# plt.show()

# for i in range(len(dataset)):
#     data = dataset[i] 
#     # points = pd.DataFrame(data.pos.numpy(), columns=['x', 'y', 'z'])
#     # cloud = PyntCloud(points)
#     # cloud.to_file("resources/clouds/TOSCA/"+str(i)+".ply")
#     if data.pos.shape[0]>40000:
#         print(i, data.pos.shape[0])





