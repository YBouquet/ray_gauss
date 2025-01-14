//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "triangle.h"
#include "helpers.h"

#include "vec_math.h"

extern "C" {
__constant__ Params params;
}

__forceinline__ __device__ void quaternion_to_matrix(const float4& q, float3& col0, float3& col1, float3& col2){
	float r = q.x;
	float i = q.y;
	float j = q.z;
	float k = q.w;
    col0=make_float3(1.0f-2.0f*(j*j+k*k),
                    2.f * (i * j + r * k),
                    2.f * (i * k - r * j));
    col1=make_float3(2.f * (i * j - r * k),
                    1.0f-2.0f*(i*i+k*k),
                    2.f * (j * k + r * i));
    col2=make_float3(2.f * (i * k + r * j),
                    2.f * (j * k - r * i),
                    1.0f-2.0f*(i*i+j*j));
}

extern "C" __global__ void __intersection__gaussian()
{
    const sphere::SphereHitGroupData* hit_group_data = reinterpret_cast<sphere::SphereHitGroupData*>( optixGetSbtDataPointer() );

    const unsigned int primitive_index = optixGetPrimitiveIndex() / 2; //* two triangles for one gaussian primitive;
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_direction  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float3 ellipse_positions = hit_group_data->positions[primitive_index];
    float3 scales=hit_group_data->scales[primitive_index];
    float gaussian_density = params.densities[primitive_index];
    // float sigm_alpha = (1/(1+expf(-gaussian_density)));
    // float density_threshold = 1.0f;

    float ratio= gaussian_density/SIGMA_THRESHOLD;
    scales=scales*sqrtf(logf(ratio*ratio));

    float3 inv_scales=make_float3(1.0f/scales.x,1.0f/scales.y,1.0f/scales.z);

    float4 quaternion=hit_group_data->quaternions[primitive_index];
    float3 U_rot,V_rot,W_rot;
    quaternion_to_matrix(quaternion,U_rot,V_rot,W_rot);

    float3 M1,M2,M3;
    U_rot=U_rot*inv_scales.x;
    V_rot=V_rot*inv_scales.y;

    float denom = dot(W_rot, ray_direction);
    if (denom > -1e-6f) {
        float t = -((ellipse_positions - ray_origin).dot(W_rot)) / denom;
        if (t >= 0)
        {
            float3 intersection_world = ray_origin + t * ray_direction;
            float3 intersection_ellipse = intersection_world - ellipse_positions;
            float u = dot(intersection_ellipse, U_rot);
            float v = dot(intersection_ellipse, V_rot);
            optixReportIntersection(
                t,
                0,
                __float_as_uint(u),
                __float_as_uint(v));
        }
    }
}


static __forceinline__ __device__ void setPayload( float3 p)
{
    optixSetPayload_0(__float_as_int( p.x ) );
    optixSetPayload_1(__float_as_int( p.y ) );
    optixSetPayload_2(__float_as_uint( p.z ) );
}


static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay( make_uint3( idx.x, idx.y, 0 ), dim, ray_origin, ray_direction );

    const float3 bbox_min = params.bbox_min;
    const float3 bbox_max = params.bbox_max;

    float3 t0,t1,tmin,tmax;
    t0 = (bbox_min - ray_origin) / ray_direction;
    t1 = (bbox_max - ray_origin) / ray_direction;
    tmin = fminf(t0, t1);
    tmax = fmaxf(t0, t1);
    float tenter = fmaxf(0.0f,fmaxf(fmaxf(tmin.x, tmin.y), tmin.z));
    float texit = fminf(tmax.x, fminf(tmax.y, tmax.z));
    const float dt = DT;
    const float slab_spacing = dt*BUFFER_SIZE;
    float transmittance = 1.0f;
    float3 ray_color = make_float3(0.0f);

    if(tenter<texit){
        float tbuffer = tenter;
        float t_min_slab, t_max_slab;
        unsigned int p0;
        unsigned int bool_not_access;

        while(tbuffer<texit && transmittance > TRANSMITTANCE_EPSILON){
            p0=0;
            t_min_slab = tbuffer;
            t_max_slab = tbuffer + slab_spacing;
            if (t_max_slab > tenter) {
                // Trace the ray against our scene hierarchy
                float buffer[BUFFER_SIZE*4]={0.0f};
                float3 result = make_float3( 0 );
                unsigned int p0, p1, p2;
                packPointer(buffer, p0, p1);
                optixTrace(
                        params.handle,
                        ray_origin,
                        ray_direction,
                        0.0f,                // Min intersection distance
                        1e16f,               // Max intersection distance
                        0.0f,                // rayTime -- used for motion blur
                        OptixVisibilityMask( 255 ), // Specify always visible
                        OPTIX_RAY_FLAG_NONE,
                        0,                   // SBT offset   -- See SBT discussion
                        1,                   // SBT stride   -- See SBT discussion
                        0,                   // missSBTIndex -- See SBT discussion
                        p0, p1, p2 );
                
                if (p0==0) {
                    tbuffer+=slab_spacing;
                    continue;
                }

                params.number_of_gaussians_per_ray[idx.x]+=p0;


            }
        }
    }
}

extern "C" __global__ void __miss__ms()
{

}


extern "C" __global__ void __anyhit__ah() {
    const unsigned int num_primitives = optixGetPayload_0();

    if (num_primitives >= params.max_prim_slice) {
        //printf("The number of primitives is greater than the maximum number of spheres per ray\n");
        optixTerminateRay();
        return;
    }

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const unsigned int idx_ray= idx.x;
    const unsigned int current_gaussian_idx = optixGetPrimitiveIndex() / 2;

    unsigned int p1,p2;
    p1=optixGetPayload_1();
    p2=optixGetPayload_2();
    float* buffer=reinterpret_cast<float*>(unpackPointer(p1,p2));

    float u = __uint_as_float(optixGetAttribute_0());
    float v = __uint_as_float(optixGetAttribute_1());
    float3 gaussian_color = make_float3(params.color_features[current_gaussian_idx*3],params.color_features[current_gaussian_idx*3+1],params.color_features[current_gaussian_idx*3+2]);

    float weight=expf(-0.5f*(u*u+v*v));
    float weight_density=weight;//*density;
    if (weight_density> SIGMA_THRESHOLD) {
        buffer[0]+=weight_density;
        buffer[1]+=gaussian_color.x*weight_density;
        buffer[2]+=gaussian_color.y*weight_density;
        buffer[3]+=gaussian_color.z*weight_density;
    }

    optixSetPayload_0(num_primitives + 1);
    optixIgnoreIntersection();
}