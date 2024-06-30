import numpy as np
import pyvista as pv
import panel as pn

# 启动虚拟帧缓冲
pv.start_xvfb()

# 确保 Panel 的 VTK 扩展被正确加载
pn.extension('vtk')

# 从 .npy 文件加载点云位置和颜色数据
pts_world = np.load('pcd_points.npy')  # 替换 'pts_world.npy' 为你的文件路径
pts_color = np.load('pcd_colors.npy')  # 替换 'pts_color.npy' 为你的文件路径

# 如果 pts_world 是 (3, N)，需要转置为 (N, 3)
if pts_world.shape[0] == 3:
    pts_world = pts_world.T

# 将 numpy 数组转换为 PyVista 的 PolyData
point_cloud = pv.PolyData(pts_world)
point_cloud['colors'] = pts_color  # 添加颜色信息

# 创建一个随机布尔掩码，75% 的概率为 True, 防止过大无法render
mask = np.random.rand(point_cloud.n_points) < 0.75
decimated_cloud = point_cloud.extract_points(mask)

# 创建 PyVista plotter
plotter = pv.Plotter(notebook=False)
#plotter.add_points(point_cloud, rgb=True, point_size=5)  # 确保使用 rgb=True 来正确显示颜色
plotter.add_points(decimated_cloud, rgb=True, point_size=2)

# 使用 Panel 显示 PyVista 的渲染窗口
panel = pn.panel(plotter.ren_win, sizing_mode='stretch_width')
#panel = pn.panel(plotter.ren_win, sizing_mode='stretch_width', width=800, height=800)


# 使用 pn.serve 来启动服务，同时允许特定 WebSocket 域的访问
pn.serve(panel, port=43857, show=True, localhost=False, 
         allow_websocket_origin=[""])