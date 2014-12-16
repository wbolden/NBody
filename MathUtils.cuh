#ifndef MATHUTILS_CUH
#define MATHUTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>


inline __device__ float3 operator*(const float &b, const float3 &a)
{
	return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float3 operator*(const float3 &a,const float &b)
{
	return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float3 operator/(const float3 &a,const float &b)
{
	return make_float3(a.x/b, a.y/b, a.z/b);
}


inline __device__ float3 operator+(const float3 &a,const float3 &b)
{
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float3 operator-(const float3 &a,const float3 &b)
{
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}


inline __device__ float3 operator-(const float4 &a,const float3 &b)
{
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}


inline __device__ float magnitude(float3& i)
{
	return sqrtf(i.x*i.x + i.y*i.y + i.z*i.z);
}

inline __device__ float dot(float3& i)
{
	return i.x*i.x + i.y*i.y + i.z*i.z;
}

inline __device__ float imagnitude(float3& i)
{
	return rsqrtf(i.x*i.x + i.y*i.y + i.z*i.z);
}

#endif