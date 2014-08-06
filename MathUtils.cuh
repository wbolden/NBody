#ifndef MATHUTILS_CUH
#define MATHUTILS_CUH

inline __device__ float magnitude(float3& i, float3& j)
{
	return sqrtf(i.x*j.x + i.y*j.y + i.z*j.z);
}


inline __device__ float fmagnitude(float3& i, float3& j)
{
	return __fsqrt_rn(__fmaf_rn(i.x, j.x, __fmaf_rn(i.y, j.y, __fmul_rn(i.z, j.z))));
}


inline __device__ float imagnitude(float3& i, float3& j)
{
	return rsqrtf(i.x*j.x + i.y*j.y + i.z*j.z);
}


inline __device__ float ifmagnitude(float3& i, float3& j)
{
	return rsqrtf(__fmaf_rn(i.x, j.x, __fmaf_rn(i.y, j.y, __fmul_rn(i.z, j.z))));
}


#endif