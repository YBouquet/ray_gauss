import os
import optix as ox
import cupy as cp
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.dlpack import from_dlpack

script_dir = os.path.dirname(__file__)
cuda_src = os.path.join(script_dir, "cuda", "triangle.cu")

img_size = (1024, 768)
SIGMA_THRESHOLD = 0.1

def quaternion_to_rotation(quaternion):
    # quaternion = [w, x, y, z]
    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]
    L1=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L2=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L3=cp.zeros((quaternion.shape[0],3),dtype=cp.float32)
    L1[:,0],L1[:,1],L1[:,2]=1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)
    L2[:,0],L2[:,1],L2[:,2]=2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)
    L3[:,0],L3[:,1],L3[:,2]=2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)
    return L1,L2,L3

def compute_2dgs_bbox(centers,scales,L1,L2,L3,densities):
    delta=cp.log((densities/SIGMA_THRESHOLD)**2)
    delta[delta<0]=0
    delta=cp.sqrt(delta)
    delta[delta>0]=1 #do not want to consider delta for now
    
    out = cp.ones((centers.shape[0],6, 3), dtype=cp.float32) * centers[:, None, :]
    scales_L1 = delta*scales * L1
    scales_L2 = delta*scales * L2
    #scales_L3 = cp.linalg.norm(delta[:,None]*scales * L3, axis=1, keepdims=False)
    
    #I want a Nx3 array with [L1,L2,L3] for each ellipsoid
    out[:, [0,1,3]] += scales_L1[:, None, :]
    out[:, [2,4,5]] -= scales_L1[:, None, :]
    out[:, [1,2,4]] += scales_L2[:, None, :]
    out[:,[0,3,5]] -= scales_L2[:, None, :]
    return out.reshape(-1,3)

# use a regular function for logging
def log_callback(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))
    pass


def create_acceleration_structure(ctx, vertices):
    build_input = ox.BuildInputTriangleArray([vertices], flags=[ox.GeometryFlags.NONE])
    gas = ox.AccelerationStructure(ctx, build_input, compact=True)
    return gas


def create_module(ctx, pipeline_opts):
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.FULL, opt_level=ox.CompileOptimizationLevel.LEVEL_0)
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module


def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_CH="__closesthit__ch")

    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=ox.CompileDebugLevel.FULL)

    pipeline = ox.Pipeline(ctx,
                           compile_options=pipeline_options,
                           link_options=link_opts,
                           program_groups=program_grps)

    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 1)  # max_dc_depth

    return pipeline


def create_sbt(program_grps, positions, scales, quaternions):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp, names=('rgb',), formats=('3f4',))
    miss_sbt['rgb'] = [0., 0., 0.]
    """
    hit_sbt = ox.SbtRecord(hit_grp, names=('positions','scales','quaternions'), formats=('u8','u8','u8'))
    hit_sbt['positions'] = positions.data.ptr
    hit_sbt['scales'] = scales.data.ptr
    hit_sbt['quaternions'] = quaternions.data.ptr
    """
    hit_sbt = ox.SbtRecord(hit_grp)
    hit_sbt = ox.SbtRecord(hit_grp, names=('positions','scales','quaternions'), formats=('u8','u8','u8'))
    hit_sbt['positions'] = positions.data.ptr
    hit_sbt['scales'] = scales.data.ptr
    hit_sbt['quaternions'] = quaternions.data.ptr
    
    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

    return sbt


def reduce_supersampling(target_width,tartget_height,ray_colors,factor):
    """ Reduce the supersampled image by a factor k
    """
    factor_x,factor_y=factor
    #Sum each factor_x values
    ray_colors_sum=ray_colors[0::factor_x]
    for i in range(1,factor_x):
        ray_colors_sum+=ray_colors[i::factor_x]
    ray_colors_sum=ray_colors_sum.reshape((tartget_height*factor_y,target_width,3))
    #Sum each factor_y values
    ray_colors_mean=ray_colors_sum[0::factor_y]
    for i in range(1,factor_y):
        ray_colors_mean+=ray_colors_sum[i::factor_y]
    return ray_colors_mean/(factor_x*factor_y)

def launch_pipeline(pipeline : ox.Pipeline, sbt, gas, colors, positions, scales, quaternions, supersampling = (1,1)):
    ray_size=(img_size[1]*img_size[0]*supersampling[0]*supersampling[1],)
    params_tmp = [
        ( 'u4', 'image_width'),
        ( 'u4', 'image_height'),
        ( '3f4', 'cam_eye'),
        ( '3f4', 'cam_U'),
        ( '3f4', 'cam_V'),
        ( '3f4', 'cam_W'),
        ( 'u8', 'color_features'),
        ( 'u8', 'positions'),
        ( 'u8', 'scales'),
        ( 'u8', 'quaternions'),
        ( 'u8', 'ray_colors'),
        ( 'u8', 'trav_handle'),
    ]

    params = ox.LaunchParamsRecord(names=[p[1] for p in params_tmp],
                                   formats=[p[0] for p in params_tmp])
    params['image_width'] = img_size[0]
    params['image_height'] = img_size[1]
    params['cam_eye'] = [0, 0, 2.0]
    params['cam_U'] = [1.10457, 0, 0]
    params['cam_V'] = [0, 0.828427, 0]
    params['cam_W'] = [0, 0, -2.0]
    params['trav_handle'] = gas.handle
    params['color_features'] = colors.data.ptr
    params['positions'] = positions.data.ptr
    params['scales'] = scales.data.ptr
    params['quaternions'] = quaternions.data.ptr
    ray_colors = cp.zeros((ray_size[0],3), dtype=cp.float32)
    params['ray_colors'] = ray_colors.data.ptr
    
    stream = cp.cuda.Stream()

    pipeline.launch(sbt, dimensions=img_size, params=params, stream=stream)

    stream.synchronize()

    return ray_colors

def torch2cupy(*args):
    # return [cp.fromDlpack(to_dlpack(x)) for x in args]
    return [cp.from_dlpack(x.detach()) for x in args]

if __name__ == "__main__":
    ctx = ox.DeviceContext(validation_mode=True, log_callback_function=log_callback, log_callback_level=3)
    pcd_position = cp.array([[0.0, 0.0, 0.0],], dtype=cp.float32)
    pcd_scale = cp.array([[0.1],], dtype=cp.float32)
    pcd_quaternion = cp.array([[1.0, 0.0, 0.0, 0.0],], dtype=cp.float32)
    pcd_density = cp.array([[1.0],], dtype=cp.float32)
    L1,L2,L3 = quaternion_to_rotation(pcd_quaternion)
    vertices = compute_2dgs_bbox(pcd_position,pcd_scale,L1,L2,L3,pcd_density)
    #vertices = cp.array([[-0.5, -0.5, 0.0],
    #                     [ 0.5, -0.5, 0.0],
    #                     [-0.5,  0.5, 0.0],
    #                     [ 0.5, -0.5, 0.0],
    #                     [ 0.5,  0.5, 0.0],
    #                     [ 0.0,  0.0, 0.0],
    #                     [ 0.0, 0.0, 0.0],
    #                     [ 0.5,  0.5, 0.0],
    #                     [-0.5,  0.5, 0.0]], dtype=np.float32)
    gas = create_acceleration_structure(ctx, vertices)
    pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                 num_payload_values=3,
                                                 num_attribute_values=3,
                                                 exception_flags=ox.ExceptionFlags.NONE,
                                                 pipeline_launch_params_variable_name="params")

    module = create_module(ctx, pipeline_options)
    program_grps = create_program_groups(ctx, module)
    pipeline = create_pipeline(ctx, program_grps, pipeline_options)
    sbt = create_sbt(program_grps)
    colors = torch2cupy(torch.ones(3).float().cuda().contiguous())[0]
    ray_colors = launch_pipeline(pipeline, sbt, gas, pcd_position, pcd_scale, colors)
    ray_colors=from_dlpack(ray_colors.toDlpack())
    ray_colors_mean= reduce_supersampling(img_size[0],img_size[1],ray_colors,(1,1))

    ray_colors_numpy = ray_colors_mean.detach().cpu().numpy().clip(0,1)
    img = ImageOps.flip(Image.fromarray(ray_colors_numpy, 'RGB'))
    img.show()