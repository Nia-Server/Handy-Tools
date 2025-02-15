import pymeshlab
import numpy as np

# 创建 MeshSet 并加载 OBJ 模型
ms = pymeshlab.MeshSet()
ms.load_new_mesh('input/test.obj')

# 设置缩放比例，例如缩放为原始尺寸的 50%
scale_factor = 30

# 应用缩放滤镜
ms.apply_filter('compute_matrix_from_scaling_or_normalization',
                axisx=scale_factor,
                axisy=scale_factor,
                axisz=scale_factor,
                uniformflag=False,  # 设置为 False 以分别指定各轴的缩放比例
                scalecenter='barycenter',  # 以模型重心为缩放中心
                freeze=True)  # 应用变换，实际改变顶点坐标

# 获取当前网格，计算目标面数（例如，保留原始面数的 90%）
mesh = ms.current_mesh()
target_face_num = int(mesh.face_number() * 0.1)

# 应用带纹理保留的减面滤镜，同时保留法线
ms.apply_filter('meshing_decimation_quadric_edge_collapse_with_texture',
                targetfacenum=target_face_num,
                preservenormal=True)

# 导出简化后的 OBJ 模型
ms.save_current_mesh('output/test.obj')
