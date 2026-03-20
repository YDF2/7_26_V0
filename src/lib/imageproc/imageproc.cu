#include <cuda_runtime.h>
#include <cstdint>

__device__ unsigned char rgb_bound(int v)
{
    return v>255?255:(v<0?0:v);
}

template<typename T>
__device__ T max(T v1, T v2)
{
    return v1>v2?v1:v2;
}

template<typename T>
__device__ T min(T v1, T v2)
{
    return v1<v2?v1:v2;
}

__device__ bool is_equal(float v1, float v2)
{
    return abs(v1-v2)<1E-6;
}

__device__ void bgr2hsv(float b, float g, float r, float *hsv)
{
    float rgbMax = max(max(r,g), b);
    float rgbMin = min(min(r,g), b);
    float delta = rgbMax-rgbMin;

    float hue, sat, val;

    val = rgbMax;
    if(is_equal(rgbMax, 0)) sat = 0;
    else sat = delta/rgbMax;

    if(is_equal(delta, 0)) hue = 0;
    else
    {
        if(is_equal(rgbMax, r))
        {
            if(g>=b) hue = 60*(g-b)/delta;
            else hue = 60*(g-b)/delta+360;
        }
        else if(is_equal(rgbMax, g))
        {
            hue = 60*(b-r)/delta+120;
        }
        else
        {
            hue = 60*(r-g)/delta+240;
        }
    }

    hsv[0] = hue;
    hsv[1] = sat;
    hsv[2] = val;
}

__global__ void yuyv2yuv_kernal(unsigned char *in, unsigned char *out, int w, int h)
{
    int x=blockIdx.x;
    int y=threadIdx.x;
    int tmp = y*w;
    int src_offset = x*2;
    int dst_offset = x*3;

    out[tmp*3+dst_offset+0] = in[tmp*2+src_offset+0];
    out[tmp*3+dst_offset+1] = in[tmp*2+src_offset+(int)powf(-1, x&1)];
    out[tmp*3+dst_offset+2] = in[tmp*2+src_offset+2+(int)powf(-1, x&1)];
}

__global__ void yuyv2bgr_kernal(unsigned char *in, unsigned char *out, int w, int h)
{
    int x=blockIdx.x;
    int y=threadIdx.x;
    int tmp = y*w;
    int src_offset = x*2;
    int dst_offset = x*3;

    unsigned char Y = in[tmp*2+src_offset+0];
    unsigned char U = in[tmp*2+src_offset+(int)powf(-1, x&1)];
    unsigned char V = in[tmp*2+src_offset+2+(int)powf(-1, x&1)];
    float r,g,b;

    r = (1.164 * (Y - 16)) + (2.018 * (V - 128));
    g = (1.164 * (Y - 16)) - (0.813 * (U - 128)) - (0.391 * (V - 128));
    b = (1.164 * (Y - 16)) + (1.596 * (U - 128));

    out[tmp*3+dst_offset+2] = rgb_bound(r);
    out[tmp*3+dst_offset+1] = rgb_bound(g);
    out[tmp*3+dst_offset+0] = rgb_bound(b);
}

__global__ void bgr2rgbfp_kernal(unsigned char *in, float *rgbfp, int w, int h)
{
    int x=blockIdx.x;
    int y=threadIdx.x;
    int offset = y*w*3+x*3;
    float rf, gf, bf;
    rf = in[offset+2]/255.0f;
    gf = in[offset+1]/255.0f;
    bf = in[offset+0]/255.0f;
    int planesize = w*h;
    int tmp = y*w+x;
    rgbfp[tmp] = rf;
    rgbfp[planesize+tmp] = gf;
    rgbfp[planesize*2+tmp] = bf;
}

__global__ void bgr2yuv422_kernal(unsigned char *in, unsigned char *out, int w, int h)
{
    int x = blockIdx.x;
    int y = threadIdx.x;
    int in_tmp = y*w*3;
    int out_tmp = y*w*2;
    int src_offset = x*6;
    int dst_offset = x*4;
    float b1 = in[in_tmp+src_offset];
    float g1 = in[in_tmp+src_offset+1];
    float r1 = in[in_tmp+src_offset+2];
    float b2 = in[in_tmp+src_offset+3];
    float g2 = in[in_tmp+src_offset+4];
    float r2 = in[in_tmp+src_offset+5];

    out[out_tmp+dst_offset] = (unsigned char)(int)(0.299*r1+0.587*g1+0.114*b1);
    out[out_tmp+dst_offset+1] = (unsigned char)(int)(-0.169*r1-0.331*g1+0.499*b1+128);
    out[out_tmp+dst_offset+2] = (unsigned char)(int)(0.299*r2+0.587*g2+0.114*b2);
    out[out_tmp+dst_offset+3] = (unsigned char)(int)(0.498*r2-0.419*g2-0.0813*b2+128);
}

__global__ void bgr2hsv_kernal(unsigned char *bgr, unsigned char *hsv, int w, int h)
{
    int x = blockIdx.x;
    int y = threadIdx.x;
    int offset = y*w*3+x*3;
    float r,g,b;
    r = bgr[offset+2];
    g = bgr[offset+1];
    b = bgr[offset+0];
    float hsv_t[3];
    bgr2hsv(b, g, r, hsv_t);
    hsv[offset+0] = rgb_bound(hsv[0]*255/360);
    hsv[offset+1] = rgb_bound(hsv[1]*255);
    hsv[offset+2] = rgb_bound(hsv[2]*255);
}

__global__ void baygr2bgr_kernal(unsigned char *bayergr, unsigned char *bgr, int w, int h,
    float ds, float rgain, float ggain, float bgain)
{
    int x = blockIdx.x;
    int y = threadIdx.x;
    int outy = h-y;
    int outx = w-x;
    float r,g,b;
    float hue, sat, val;
    float rn, gn, bn;

    b = bayergr[(y+((y+1)&1))*w+x-(x&1)]*bgain;
    g = bayergr[y*w+x-(x&1)+(y&1)]*ggain;
    r = bayergr[(y-(y&1))*w+x+((x+1)&1)]*rgain;

    float hsv[3];
    bgr2hsv(b,g,r, hsv);
    hue = hsv[0];
    sat = hsv[1];
    val = hsv[2];
    if(ds>=0) sat = sat+(1-sat)*ds;
    else sat = sat+sat*ds;

    int hi = hue/60.0;
    float f = hue/60.0-hi;
    float p = val*(1-sat);
    float q = val*(1-f*sat);
    float t = val*(1-(1-f)*sat);
    switch(hi)
    {
        case 0:
            rn = val; gn = t; bn = p; break;
        case 1:
            rn = q; gn = val; bn = p; break;
        case 2:
            rn = p; gn = val; bn = t; break;
        case 3:
            rn = p; gn = q; bn = val; break;
        case 4:
            rn = t; gn = p; bn = val; break;
        case 5:
            rn = val; gn = p; bn = q; break;
        default:
            break;
    }
    bgr[outy*w*3+outx*3+0] = rgb_bound(bn);
    bgr[outy*w*3+outx*3+1] = rgb_bound(gn);
    bgr[outy*w*3+outx*3+2] = rgb_bound(rn);
}

template<typename T>
__global__ void resize_packed_kernal(T *in, int iw, int ih, T *out, int ow, int oh)
{
    int x = blockIdx.x;
    int y = threadIdx.x;
    int offset_out = y*ow*3+x*3;
    float h_scale_rate = (float)ih/oh;
    float w_scale_rate = (float)iw/ow;
    float y_scale = h_scale_rate * y;
    float x_scale = w_scale_rate * x;
    int j = y_scale, i = x_scale;
    float u = y_scale-j, v = x_scale-i;
    int offset_in1 = j*iw*3;
    int offset_in2 = (j+1)*iw*3;
    if(j+1>=ih || i+1>=iw)
    {
        out[offset_out+0] = in[offset_in1+i*3];
        out[offset_out+1] = in[offset_in1+i*3+1];
        out[offset_out+2] = in[offset_in1+i*3+2];
    }
    else
    {
        unsigned char x1,x2,x3,x4;
        x1 = in[offset_in1+i*3];
        x2 = in[offset_in1+(i+1)*3];
        x3 = in[offset_in2+i*3];
        x4 = in[offset_in2+(i+1)*3];
        out[offset_out+0] = ((1-u)*(1-v)*x1+(1-u)*v*x2+u*(1-v)*x3+u*v*x4);
        x1 = in[offset_in1+i*3+1];
        x2 = in[offset_in1+(i+1)*3+1];
        x3 = in[offset_in2+i*3+1];
        x4 = in[offset_in2+(i+1)*3+1];
        out[offset_out+1] = ((1-u)*(1-v)*x1+(1-u)*v*x2+u*(1-v)*x3+u*v*x4);
        x1 = in[offset_in1+i*3+2];
        x2 = in[offset_in1+(i+1)*3+2];
        x3 = in[offset_in2+i*3+2];
        x4 = in[offset_in2+(i+1)*3+2];
        out[offset_out+2] = ((1-u)*(1-v)*x1+(1-u)*v*x2+u*(1-v)*x3+u*v*x4);
    }
}

__global__ void build_map_kernal(float *pCamK, float *pDistort, float *pInvNewCamK, float *pMapx, float *pMapy, int outImgW, int outImgH)
{
	const int tidx = blockDim.x*blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y*blockIdx.y + threadIdx.y;
	if (tidx < outImgW && tidy < outImgH)
	{
		float k1 = pDistort[0];
		float k2 = pDistort[1];
		float p1 = pDistort[2];
		float p2 = pDistort[3];
		float k3, k4, k5, k6, s1, s2, s3, s4;
		k3 = k4 = k5 = k6 = s1 = s2 = s3 = s4 = 0;
		float fx = pCamK[0];
		float fy = pCamK[4];
		float u0 = pCamK[2];
		float v0 = pCamK[5];

		float _x = tidx*pInvNewCamK[0] + tidy*pInvNewCamK[1] + pInvNewCamK[2];
		float _y = tidx*pInvNewCamK[3] + tidy*pInvNewCamK[4] + pInvNewCamK[5];
        float _w = tidx*pInvNewCamK[6] + tidy*pInvNewCamK[7] + pInvNewCamK[8];
        
		float w = 1. / _w;
		float x = _x * w;
		float y = _y * w;

		float x2 = x*x;
		float y2 = y*y;
		float r2 = x2 + y2;
		float _2xy = 2 * x*y;
		float kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2) / (1 + ((k6*r2 + k5)*r2 + k4)*r2);
		float xd = (x*kr + p1*_2xy + p2*(r2 + 2 * x2) + s1*r2 + s2*r2*r2);
		float yd = (y*kr + p1*(r2 + 2 * y2) + p2*_2xy + s3*r2 + s4*r2*r2);

		float invProj = 1.;
		float u = fx*invProj*xd + u0;
		float v = fy*invProj*yd + v0;

		int mapIdx = tidy*outImgW + tidx;
		pMapx[mapIdx] = (float)u;
		pMapy[mapIdx] = (float)v;
	}
}

__global__ void remap_kernal(unsigned char* pSrcImg, unsigned char* pDstImg, float* pMapx, float* pMapy, int inWidth, int inHeight, 
        int outWidth, int outHeight, int channels)
{
	const int tidx = blockDim.x*blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y*blockIdx.y + threadIdx.y;
	if (tidx < outWidth && tidy < outHeight)
	{
		int mapIdx = tidy*outWidth + tidx;
		float u = pMapx[mapIdx];
		float v = pMapy[mapIdx];

		int u1 = floor(u);
		int v1 = floor(v);
		int u2 = u1 + 1;
		int v2 = v1 + 1;
		if (u1 >= 0 && v1 >= 0 && u2 < inWidth && v2 < inHeight)
		{
			float dx = u - u1;
			float dy = v - v1;
			float weight1 = (1 - dx)*(1 - dy);
			float weight2 = dx*(1 - dy);
			float weight3 = (1 - dx)*dy;
			float weight4 = dx*dy;

			int resultIdx = mapIdx * 3;
			for (int chan = 0; chan < channels; chan++)
			{
				pDstImg[resultIdx + chan] = (unsigned char)(weight1*pSrcImg[(v1*inWidth + u1) * 3 + chan]
					+ weight2*pSrcImg[(v1*inWidth + u2) * 3 + chan]
					+ weight3*pSrcImg[(v2*inWidth + u1) * 3 + chan]
					+ weight4*pSrcImg[(v2*inWidth + u2) * 3 + chan]);
			}
		}
	}
}

namespace imgproc
{
    void cudaYUYV2YUV(unsigned char *in, unsigned char *out, int w, int h)
    {
        yuyv2yuv_kernal<<<w, h>>>(in,out,w,h);
    }

    void cudaYUYV2BGR(unsigned char *in, unsigned char *out, int w, int h)
    {
        yuyv2bgr_kernal<<<w, h>>>(in,out,w,h);
    }

    void cudaBayer2BGR(unsigned char *bayer, unsigned char *bgr, int w, int h, 
        float sat, float rgain, float ggain, float bgain)
    {
        baygr2bgr_kernal<<<w,h>>>(bayer, bgr, w, h, sat, rgain, ggain, bgain);
    }

    void cudaBGR2RGBfp(unsigned char *bgr, float *rgbfp, int w, int h)
    {
        bgr2rgbfp_kernal<<<w,h>>>(bgr, rgbfp, w, h);
    }

    void cudaBGR2YUV422(unsigned char *bgr, unsigned char *yuv422, int w, int h)
    {
        bgr2yuv422_kernal<<<w/2, h>>>(bgr, yuv422, w, h);
    }
    
    void cudaBGR2HSV(unsigned char *bgr, unsigned char *hsv, int w, int h)
    {
        bgr2hsv_kernal<<<w, h>>>(bgr, hsv, w, h);
    }

    void cudaResizePacked(float *in, int iw, int ih, float *sized, int ow, int oh)
    {
        resize_packed_kernal<<<ow, oh>>>(in, iw, ih, sized, ow, oh);
    }

    void cudaResizePacked(unsigned char *in, int iw, int ih, unsigned char *sized, int ow, int oh)
    {
        resize_packed_kernal<<<ow, oh>>>(in, iw, ih, sized, ow, oh);
    }

    void cudaUndistored(unsigned char *in, unsigned char *out, float *pCamK, float *pDistort, float *pInvNewCamK, 
        float* pMapx, float* pMapy, int w, int h, int c)
    {
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
        build_map_kernal <<<grid, block >>> (pCamK, pDistort, pInvNewCamK, pMapx, pMapy, w, h);
        cudaThreadSynchronize();
        remap_kernal <<<grid, block >>> (in, out, pMapx, pMapy, w, h, w, h, c);
        cudaThreadSynchronize();
    }
};
