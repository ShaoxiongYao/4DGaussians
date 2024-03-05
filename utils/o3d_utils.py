import open3d as o3d

def set_view_params(o3d_vis, view_params={}):
    ctr = o3d_vis.get_view_control()
    if "zoom" in view_params.keys():
        ctr.set_zoom(view_params["zoom"])
    if "front" in view_params.keys():
        ctr.set_front(view_params["front"])
    if "lookat" in view_params.keys():
        ctr.set_lookat(view_params["lookat"])
    if "up" in view_params.keys():
        ctr.set_up(view_params["up"])
