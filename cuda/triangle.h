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
#define MAX_CLOSEST_HITS 16

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
    unsigned int            image_width;
    unsigned int            image_height;
    float3                  cam_eye;
    float3                  cam_u, cam_v, cam_w;
    float*                  color_features;
    float3*                 positions;
    float3*                 scales;
    float4*                 quaternions;
    float3*                 ray_colors;
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

struct HitInfo {      
    float4 color;      // Color at the hit point
    float t;       // Depth along the ray
};

struct HitInfoNode {
    HitInfo* info;              // HitInfo data
    HitInfoNode* next;        // Pointer to the next node in the linked list
};

static __forceinline__ __device__ HitInfoNode* initHitInfoNode() {
    HitInfoNode* node = new HitInfoNode(); // Allocate memory for a new node
    node->info = nullptr;                   // Initialize info pointer
    node->next = nullptr;                   // Initialize next pointer
    return node;
}

// Function to add a HitInfoNode to the linked list in increasing order of t
static __forceinline__ __device__ void addHitInfoNode(HitInfoNode** head, HitInfo* hitInfo) {
    HitInfoNode* newNode = initHitInfoNode(); // Create a new node
    newNode->info = hitInfo;                   // Set the hit info

    // If the list is empty or the new node should be the new head
    if (*head == nullptr || newNode->info->t < (*head)->info->t) {
        newNode->next = *head;                 // Point new node to the current head
        *head = newNode;                       // Update head to the new node
        return;
    }

    // Traverse the list to find the correct position
    HitInfoNode* current = *head;
    while (current->next != nullptr && newNode->info->t >= current->next->info->t) {
        current = current->next;               // Move to the next node
    }

    // Insert the new node in the correct position
    newNode->next = current->next;            // Link new node to the next node
    current->next = newNode;                   // Link current node to the new node
}

static __forceinline__ __device__ void deleteHitInfoList(HitInfoNode** head) {
    HitInfoNode* current = *head; // Start from the head
    while (current != nullptr) {
        HitInfoNode* nextNode = current->next; // Store the next node
        //If HitInfo was dynamically allocated, free it here
        if (current->info != nullptr) {
            delete current->info; //Uncomment if HitInfo is dynamically allocated
        }
        delete current; // Free the current node
        current = nextNode; // Move to the next node
    }
    head = nullptr; // Set head to nullptr after deletion
}


struct KBuffer {
    HitInfo hits[MAX_CLOSEST_HITS];  // Using MAX_CLOSEST_HITS (16) as defined
    int count;                       // Current number of hits

    __device__ int binary_search(float t) {
        int left = 0;
        int right = count - 1;
        
        // Handle edge cases first
        if (count == 0 || t < hits[0].t) return 0;
        if (t >= hits[right].t) return count;
        
        // Binary search
        while (left < right) {
            int mid = (left + right) / 2;
            
            if (hits[mid].t == t) {
                return mid;
            } else if (hits[mid].t < t) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return left;
    }

    __device__ void insert(const HitInfo& hit) {
        if (count == 0) {
            hits[0] = hit;
            count = 1;
            return;
        }
        
        // Find insertion position (binary search could be used for larger k)
        int pos = binary_search(hit.t);
        
        // If we're at the end and buffer is full, ignore if farther than last element
        if (pos >= MAX_CLOSEST_HITS) return;
        
        // Shift elements to make room
        int shift_end = min(count, MAX_CLOSEST_HITS - 1);
        for (int i = shift_end; i > pos; i--) {
            hits[i] = hits[i-1];
        }
        
        // Insert new hit
        hits[pos] = hit;
        count = min(count + 1, MAX_CLOSEST_HITS);
    }
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

