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

#pragma once

#define BUFFER_SIZE 8
#define DT 0.0025f
#define TRANSMITTANCE_EPSILON 0.003f
#define SIGMA_THRESHOLD 0.1f

namespace sphere {
    // const unsigned int NUM_ATTRIBUTE_VALUES = 4u;

    struct SphereHitGroupData {
        float3* positions;
        float3* scales;
        float4* quaternions;
    };
}

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    float*                 color_features;
    float3*                 positions;
    float3*                 scales;
    float4*                 quaternions;
    
    OptixTraversableHandle handle;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};

static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

