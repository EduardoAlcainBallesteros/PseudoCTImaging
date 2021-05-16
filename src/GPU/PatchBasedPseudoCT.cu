/*
* pseudo-CT estimation based on multi-atlas with patches on mutlicore and manycore platforms. For further details see [1,2]
* If you use this software for research purposes, YOU MUST CITE the corresponding
* of the following papers in any resulting publication:
*
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017). https://doi.org/10.1007/s11554-016-0568-0 
* [2] Angel Torrado-Carvajal, Joaquin L. Herraiz, Eduardo Alcain, Antonio S. Montemayor, Lina Garcia-Cañamaque, Juan A. Hernandez-Tamames, Yves Rozenholc and Norberto Malpica Journal of Nuclear Medicine January 2016, 57 (1) 136-143; DOI: https://doi.org/10.2967/jnumed.115.156299
* PatchBasedPseudoCT is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PatchBasedPseudoCT is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with fastms. If not, see <http://www.gnu.org/licenses/>.
*
* Copyright 2020 Eduardo Alcain Ballesteros eduardo.alcain.ballesteros@gmail.com
*/


#include <float.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <stdio.h>
//#include <Windows.h>
#include <stdlib.h>   // For _MAX_PATH definition
#include <algorithm>
#include <stdio.h>
#include <malloc.h>
#include  <math.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include <map>

// https://docs.microsoft.com/en-us/cpp/build/reference/d-preprocessor-definitions?view=vs-2019
#ifdef MATLAB 
#pragma message( "MATLAB Compiled")
#include "mat.h"
#else 
#pragma message( "Random data")
#endif
// For the CUDA runtime routines (prefixed with "cuda_")
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"
#include <cublas_v2.h>
#include <cusparse_v2.h>
// https://stackoverflow.com/questions/19352941/trying-to-get-cuda-working-sample-cant-find-helper-cuda-h

//#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper function CUDA error checking and intialization

#include <iostream>



#include <iostream>
#include <fstream>

using namespace std;

typedef enum
{
	GM = 0,
	GM2 = 1,
	SM = 2,
	GpuTestTime=4

} EXEC_CHOICE;


typedef struct blk {
	int refX;
	int refY;
	int refZ;
	int thx;
	int thy;
	int thz;
	int x, y, z;
	int xIndex, yIndex, zIndex;
	float index;
	double sum;
	double product;
	double  sumRet, arg, weight, sum_weight, lbl;
	int ii;
	int bx, by, bz;

}block;

// Mask for the patch either in Atlas or in the image
typedef struct mask {
	int z;
	int zZ;
	int c;
	int cC;
	int r;
	int rR;
	int rows;
	int cols;
	int mats;

}matrixMask;


typedef struct d_mask {
	int z; // z inf
	int zZ; // z sup
	int c; // y inf
	int cC; // y sup
	int r; // x inf 
	int rR; // x sup


}d_matrixMask;



                           \

#define CLEANUP(s)                                   \
    do {                                                 \
    if (d_img)                 checkCudaErrors(cudaFree(d_img));             \
    if (d_atlas)                 checkCudaErrors(cudaFree(d_atlas));             \
    if (d_atlas_label)               checkCudaErrors(cudaFree(d_atlas_label));          \
	if (d_label_segmentation)              checkCudaErrors(cudaFree(d_label_segmentation));          \
	fflush (stdout);                                 \
					    } while (0)



/**
* Kernel to synthesise the new modality using global memory (GM)
* @param d_img: image MRI
* @param d_atlas: dictionary
* @param d_atlas_label: labels for each atlas in the dictionary
* @param d_label_segmentation: image pseudo-CT
* @param N: number of elements in a image nx*ny*nz (222x222x112)
* @param NN: number of elements within a patch (patch_size *patch_size*patch_size) 27;
* @param nx: dimension of x from the images (either atlas, input image, etc) 222
* @param ny: dimension of x from the images (either atlas, input image, etc) 222
* @param nz: dimension of x from the images (either atlas, input image, etc) 112
* @param div: see formula (1) 2S beta sigma^2 in [1]
* @param patch_size: size of the patch for our experiments 3 (3x3x3)
* @param half_patch: (patch_size - 1) / 2
* @param half_window: half of the neighbourhood (5 (window=11) or 4 (window=9) or 3 (window=7)) (window - 1) / 2
* @param num_atlas number of atlas in the dictionary

* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017).
https://doi.org/10.1007/s11554-016-0568-0
*/


__global__ void MultiPatchSegmentationGMKernel(const float * d_img, const float *d_atlas, const float *d_atlas_label, float *d_label_segmentation, const int N, const float NN, const int nx, const int ny, const int nz, const double div, const int patch_size,
	const int window, const int half_patch, const int half_window, const int num_atlas) {
	// Variable for the patches (corners of the patch) for input image and atlas
	d_mask maskIm, maskAtlas;
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned int ii = yIndex + ny*xIndex + nx*ny*zIndex;
	int     left_x, rigth_x, left_y, rigth_y, left_z, rigth_z;
	int left_u, rigth_u, left_v, rigth_v, left_w, rigth_w;
	double weight = 0.0f, product = 0.0f, sum_weight = 0.0f, lbl = 0.0f;
	
	double patch = 0;
	double patch_at = 0;
	double sumRet = 0;
	double arg = 0.0f;
	if (ii < N) {
		int z = ii / (nx*ny);
		int aux = (ii - (z*nx*ny));
		// ROw
		int x = aux / ny;
		// Col
		int y = aux% ny;
		// Check that we can calculate the neighbourhood
		if (z >= half_window + half_patch && z < nz - half_window - half_patch &&
			x >= half_window + half_patch && x < nx - half_window - half_patch  &&
			y >= half_window + half_patch && y < ny - half_window - half_patch
			) {
			// Calculate the corners for the patch
			left_x = x - half_patch;
			rigth_x = x + half_patch;
			left_y = y - half_patch;
			rigth_y = y + half_patch;
			left_z = z - half_patch;
			rigth_z = z + half_patch;
			// Matlab
			//patch = img(left_x:rigth_x,left_y:rigth_y,left_z:rigth_z);
			// We store the corners for the patch
			maskIm.c = left_y; maskIm.cC = rigth_y;
			maskIm.r = left_x; maskIm.rR = rigth_x;
			maskIm.z = left_z; maskIm.zZ = rigth_z;
			// Loop for the atlas
			for (int at = 0; at < num_atlas; at++) {
				// Loop for iterating through neighbourhood inside a atlas
				for (int w = z - half_window; w <= z + half_window; w++) {
					for (int u = x - half_window; u <= x + half_window; u++) {

						for (int v = y - half_window; v <= y + half_window; v++) {


							// Calculate the corners for the patch (atlas)
							left_u = u - half_patch; rigth_u = u + half_patch;
							left_v = v - half_patch; rigth_v = v + half_patch;
							left_w = w - half_patch; rigth_w = w + half_patch;
							//Matlab
							//patch_at = atlas(left_u:rigth_u,left_v:rigth_v,left_w:rigth_w);
							// Use to calculate the formula 1 ini our paper
							lbl = d_atlas_label[(at*N) + (w*nx*ny) + (u*ny) + v];
							// We store the corners in the struct
							maskAtlas.c = left_v; maskAtlas.cC = rigth_v;
							maskAtlas.r = left_u; maskAtlas.rR = rigth_u;
							maskAtlas.z = left_w; maskAtlas.zZ = rigth_w;




							//ms = (patch-patch_at).^2;
							//arg = sum(ms(:))/(2*N*beta*pow(sigma,2));
							//weight = exp(-arg);

							//weight =formula(maskIm,  maskAtlas, d_img, d_atlas, N,  sigma, beta);

							// Auxiliary Variables
							patch = 0.0f; patch_at = 0.0f;
							// SumRet contains the MSE
							sumRet = 0; arg = 0.0f; weight = 0.0f;
							// Calculate the MSE (formula 1 in our paper) between patch from the image and specific atlas 
							// Iterate through the patch
							for (int k = maskIm.z, kk = maskAtlas.z; (k <= maskIm.zZ) && (kk <= maskAtlas.zZ); k++, kk++) {

								for (int i = maskIm.r, iii = maskAtlas.r; (i <= maskIm.rR) && (iii <= maskAtlas.rR); i++, iii++) {

									for (int j = maskIm.c, jj = maskAtlas.c; (j <= maskIm.cC) && (jj <= maskAtlas.cC); j++, jj++) {
										patch = d_img[(k*nx*ny) + (i*ny) + j]; // pixel/voxel patch from the input image MRI
										patch_at = d_atlas[(at*N) + (kk*nx*ny) + (iii*ny) + jj]; //pixel/voxel patch from the atlas 
																								 // I do not know why CUDA does not like this line 
																								 //sumRet += pow(patch -patch_at,2);
										sumRet += (patch - patch_at) *(patch - patch_at);

									} // end j,jj
								} // end i,iii

							} // end k,kk 

							arg = sumRet /div;
							// Calculate the weight formula 2
							weight = exp(-arg);
							
							// Group - Wise Label Propagation (accumulative part ) formula 2 in our paper (A=product/B=sum_weight)
							// Update A multiply weight by the label of the atlas for this voxel
							product += weight * lbl;
							// Update B
							sum_weight += weight;


						}
					}
				}
			}
			// Group - Wise Label Propagation (calculation part) formula 2 in our paper (A/B)
			// if the product and sum_weight is greater than the precision
			if (abs(product) > DBL_EPSILON && abs(sum_weight) > DBL_EPSILON) {
				d_label_segmentation[ii] = (float)(product / sum_weight);
			}
			else {
				// otherwise we asssign NAN and there is a process of regularization point 2.3 of our paper
				d_label_segmentation[ii] = NAN;
			}

		}
	}

}





/**
* Kernel to synthesise the new modality using global memory (GM) with registers for the MRI input patch
* @param d_mask: for testing purposes
* @param d_img: image MRI
* @param d_atlas: dictionary
* @param d_atlas_label: labels for each atlas in the dictionary
* @param d_label_segmentation: image pseudo-CT
* @param N: number of elements in a image nx*ny*nz (222x222x112)
* @param div: see formula (1) 2S beta sigma^2 in [1]
* @param nx: dimension of x from the images (either atlas, input image, etc) 222
* @param ny: dimension of x from the images (either atlas, input image, etc) 222
* @param nz: dimension of x from the images (either atlas, input image, etc) 112
* @param patch_size: size of the patch for our experiments 3 (3x3x3)
* @param window:
* @param half_patch: (patch_size - 1) / 2
* @param half_window: half of the neighbourhood (5 (window=11) or 4 (window=9) or 3 (window=7)) (window - 1) / 2
* @param num_atlas number of atlas in the dictionary
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017).
https://doi.org/10.1007/s11554-016-0568-0
*/
__global__ void MultiPatchSegmentationGM2Kernel(block *d_mask, const float * d_img, const float *d_atlas, const float *d_atlas_label, float *d_label_segmentation, const int N, const double div, const int nx, const int ny, const int nz, const int patch_size,
	const int window, const int half_patch, const int half_window, const int num_atlas) {
	d_matrixMask maskIm, maskAtlas;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned int ii = y + ny*x + nx*ny*z;
	double weight = 0.0f, product = 0.0f, sum_weight = 0.0f, lbl = 0.0f;
	double aux = 0, sumRet = 0, arg = 0.0f;
	int ptr;
	int idxMat;
	if (ii < N) {
		
		// Check that we can calculate the neighbourhood
		if (z >= half_window + half_patch && z < nz - half_window - half_patch &&
			x >= half_window + half_patch && x < nx - half_window - half_patch  &&
			y >= half_window + half_patch && y < ny - half_window - half_patch
			) {
			// Calculate the corners for the patch
			maskIm.c = y - half_patch; maskIm.cC = y + half_patch;
			maskIm.r = x - half_patch; maskIm.rR = x + half_patch;
			maskIm.z = z - half_patch; maskIm.zZ = z + half_patch;
			// Array for the patch from the input image MRI 3x3x3= 27
			float Ims[27];
			ptr = 0;
			// Loop to insert column after column (aplanar) into Ims our patch for the input image
			// for an specific voxel determined by the thread
			for (int _k = maskIm.z; (_k <= maskIm.zZ); _k++) {

				for (int _i = maskIm.r; (_i <= maskIm.rR); _i++) {

					for (int _j = maskIm.c; (_j <= maskIm.cC); _j++) {
						// Read from global memory to register
						Ims[ptr] = d_img[(_k*nx*ny) + (_i*ny) + _j];
						ptr++;
					}// end _k
				} // end _i
			} // end _k
			  // Loop for the atlas
			for (int at = 0; at < num_atlas; at++) {
				idxMat = 0;
				// Loop for iterating through neighbourhood inside a atlas
				for (int w = z - half_window; w <= z + half_window; w++) {
					for (int u = x - half_window; u <= x + half_window; u++) {

						for (int v = y - half_window; v <= y + half_window; v++) {
							// Use to calculate the formula 1 in our paper
							lbl = d_atlas_label[(at*N) + (w*nx*ny) + (u*ny) + v];
							// We store the corners in the struct
							maskAtlas.c = v - half_patch; maskAtlas.cC = v + half_patch;
							maskAtlas.r = u - half_patch; maskAtlas.rR = u + half_patch;
							maskAtlas.z = w - half_patch; maskAtlas.zZ = w + half_patch;


							sumRet = 0; arg = 0.0f; weight = 0.0f;
							// We set the ptr each time we move inside the nh
							ptr = 0;
							// Calculate the MSE (formula 1 in our paper) between patch from the image and specific atlas 
							// Iterate through the patch
							for (int kk = maskAtlas.z; (kk <= maskAtlas.zZ); kk++) {

								for (int iii = maskAtlas.r; (iii <= maskAtlas.rR); iii++) {

									for (int jj = maskAtlas.c; (jj <= maskAtlas.cC); jj++) {
										aux = Ims[ptr] - d_atlas[(at*N) + (kk*nx*ny) + (iii*ny) + jj];
										// Pointer for iterating through the pathc from the input image
										ptr++;
										// I do not know why CUDA does not like this line 
										//sumRet += pow(patch -patch_at,2);
										sumRet += aux *aux;

									} // end jj
								} // end iii

							} // end kk
							  // div=2 * NN*beta*pow(sigma, 2) and has been calculate out of the kernel
							arg = sumRet / div;
							// Calculate the weight formula 2
							weight = exp(-arg);
							// Group - Wise Label Propagation (accumulative part ) formula 2 in our paper (A=product/B=sum_weight)
							// Update A multiply weight by the label of the atlas for this voxel
							product += weight * lbl;
							sum_weight += weight;

						

						} // end v
					} // end u
				} // end w
			}
			// Group - Wise Label Propagation (calculation part) formula 2 in our paper (A/B)
			// if the product and sum_weight is greater than the precision
			//d_label_segmentation[ii] =(float) rint(product / sum_weight);
			if (abs(product) > DBL_EPSILON && abs(sum_weight) >DBL_EPSILON)
				d_label_segmentation[ii] = (float)(product / sum_weight);
			else
				// otherwise we asssign NAN and there is a process of regularization point 2.3 of our paper
				d_label_segmentation[ii] = NAN;


		} // end Check that we can calculate the neighbourhood
	} // end ii < N
} // end kernel



  /**
  * Kernel to synthesise the new modality using shared memory (GM) with registers for the MRI input patch
  * Nh configurations for the data loads for 10x10x10 Threads in a block
  * - 7   NH=7, LIMIT=168, LOAD1=5, LOAD2=6, SM=18, SM2=324
  * - 9   NH=9, LIMIT=1000, LOAD1=8, LOAD2=0, SM=20, SM2=400
  * - 11  NH=11, LIMIT=352, LOAD1=10, LOAD2=11, SM=22, SM2=484
  * @param d_mask: for testing purposes
  * @param d_img: image MRI
  * @param d_atlas: dictionary images MRI
  * @param d_atlas_label: labels for each atlas in the dictionary
  * @param d_label_segmentation: image pseudo-CT
  * @param N: number of elements in a image nx*ny*nz (222x222x112)
  * @param div: number of elements within a patch (patch_size *patch_size*patch_size) 27;
  * @param nx: dimension of x from the images (either atlas, input image, etc) 222
  * @param ny: dimension of x from the images (either atlas, input image, etc) 222
  * @param nz: dimension of x from the images (either atlas, input image, etc) 112
  * @param patch_size: size of the patch for our experiments 3 (3x3x3)
  * @param half_patch: (patch_size - 1) / 2
  * @param half_window: half of the neighbourhood (5 (window=11) or 4 (window=9) or 3 (window=7)) (window - 1) / 2
  * @param num_atlas number of atlas in the dictionary
  * @param NH: neighbourhood size
  * @param LIMIT: number of threads before change the set.
  * @param LOAD1: number of elements to load for first set of threads
  * @param LOAD2: number of elements to load for second set of threads
  * @param SM: size of the shared memory only one side
  * @param SM2: size of the shared memory SM^2
  */

template<typename T>
__global__ void MultiPatchSegmentationSMKernel( const float * d_img, const float *d_atlas, const float *d_atlas_label, float *d_label_segmentation, const int N, const T div, const int nx, const int ny, const int nz, const int patch_size,
	const int half_patch, const int half_window, const int num_atlas, const int NH, const int LIMIT, const int LOAD1, const int LOAD2, const int SM, const int SM2) {
	
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
	unsigned int ii = y + ny*x + nx*ny*z;
	
	T weight = 0.0, product = 0.0, sum_weight = 0.0, lbl = 0.0;

	T aux;
	T sumRet = 0.0;
	T arg = 0.0;
	int indexZ, indexX, indexY, many, ptr;
	// dynamic Shared memory declaration
	extern __shared__ float Nbh_s[];

	int xxRef, yyRef, zzRef;
	// patch for the input image
	float Ims[27];
	if (ii < N) {
		// Reference where the thread is in order to load the data into shared memory
		xxRef = blockIdx.x * blockDim.x;
		yyRef = blockIdx.y * blockDim.y;
		zzRef = blockIdx.z * blockDim.z;
		xxRef = xxRef - half_patch - half_window;
		yyRef = yyRef - half_patch - half_window;
		zzRef = zzRef - half_patch - half_window;
		// Loop for the atlas
		for (short at = 0; at < num_atlas; at++) {


			// Load into Shared memory the neighbourhood
			// As the division between threads in the block and voxels/pixels in the shared memory is not exact
			// every thread has to load more than one pixel/voxel () with calculate_ptr.m we calculate
			// how many pixel/voxel a thread has to load from global memory into shared memory
			// The 168 threads will load 5 voxels/pixels
			if ((threadIdx.z * blockDim.x * blockDim.x) + (threadIdx.x * blockDim.x) + threadIdx.y < LIMIT) {
				many = LOAD1;
			}
			else {
				many = LOAD2;
			}
			// Loop to load the pixel/voxel from global memory into shared memory
			for (short th = 0; th < many; th++) {

				// we calculate the pointer inside the shared memory where we have to store the new value
				if ((threadIdx.z * blockDim.x * blockDim.x) + (threadIdx.x * blockDim.x) + threadIdx.y < LIMIT) {

					ptr = (threadIdx.y*many) + th + (blockDim.y*many) * threadIdx.x + (many*blockDim.y *blockDim.x) * threadIdx.z;
				}
				else {
					ptr = (threadIdx.y*many) - LIMIT + th + (blockDim.y*many) * threadIdx.x + (many*blockDim.y *blockDim.x) * threadIdx.z;
				}

				indexZ = (ptr / SM2) + zzRef;
				indexX = ptr / SM % SM + xxRef;
				indexY = ptr % SM + yyRef;

				// We assure that the indices are inside the limits
				if (indexX >= 0 & indexY >= 0 & indexZ >= 0 & indexX < nx &  indexY < ny & indexZ < nz) {
					// Load the data (pixel/voxel) from gloabl memory into shared memory
					Nbh_s[ptr] = d_atlas[(at*N) + (indexZ*nx*ny) + (indexX*ny) + indexY];

				}
				else {
					// otherwise is zero
					Nbh_s[ptr] = 0;



				}
				

			}// for load 
			 // Wait until all the threads have finished loading the data into shared memory
			__syncthreads();



			//if (blockIdx.x == 18 && blockIdx.y == 13 && blockIdx.z == 1){
			/*if (ii == 1292702){*/
			// Check that we can calculate the neighbourhood
			if (z >= half_window + half_patch && z < nz - half_window - half_patch &&
				x >= half_window + half_patch && x < nx - half_window - half_patch  &&
				y >= half_window + half_patch && y < ny - half_window - half_patch
				) {


				/*maskIm.c = y - half_patch; maskIm.cC = y + half_patch;
				maskIm.r = x - half_patch; maskIm.rR = x + half_patch;
				maskIm.z = z - half_patch; maskIm.zZ = z + half_patch;*/
				//maskIm.cols = ny; maskIm.rows = nx; maskIm.mats = nz;
				//int idxMat = 0;
				if (at == 0) {
					ptr = 0;
					// Loop to insert column after column (aplanar) into Ims our patch for the input image
					// for an specific voxel determined by the thread
					for (int _k = z - half_patch; (_k <= z + half_patch); _k++) {

						for (int _i = x - half_patch; (_i <= x + half_patch); _i++) {

							for (int _j = y - half_patch; (_j <= y + half_patch); _j++) {

								Ims[ptr] = d_img[(_k*nx*ny) + (_i*ny) + _j];
								ptr++;
							} // end for _j
						} // end for _i
					} // end for _k
				} // end if at==0
				  // Loop for iterating through neighbourhood inside a atlas
				  // z,x,y global position inside the grid
				  // _w (z) ,_u (x) ,_v (y) local position inside the block
				  // we start where our threadIdx.(x/y/z) are and we create the neighbourhood 7x7x7 surrounded
				for (int w = z - half_window, _w = threadIdx.z + 1; _w <= threadIdx.z + NH; _w++, w++) {
					for (int u = x - half_window, _u = threadIdx.x + 1; _u <= threadIdx.x + NH; _u++, u++) {
						for (int v = y - half_window, _v = threadIdx.y + 1; _v <= threadIdx.y + NH; _v++, v++) {

							//Matlab
							//patch_at = atlas(left_u:rigth_u,left_v:rigth_v,left_w:rigth_w);
							// Use to calculate the formula 1 in our paper
							lbl = d_atlas_label[(at*N) + (w*nx*ny) + (u*ny) + v];


							//patch = 0.0f; patch_at = 0.0f;
							sumRet = 0; weight = 0.0f;
						

							// Set the pointer for input image patch to zero
							ptr = 0;
							// Calculate the MSE (formula 1 in our paper) between patch from the image and specific atlas (shared memory Nbh_s)
							// Iterate through the patch
							for (int _kk = _w - 1; (_kk <= _w + 1); _kk++) {

								for (int _iii = _u - 1; _iii <= _u + 1; _iii++) {

									for (int _jj = _v - 1; (_jj <= _v + 1); _jj++) {

										//patch = Ims[i][j][k];
										//patch_at = Nbh_s[(_kk * 484) + (_iii * 22) + _jj];
										// with the local position _kk,_iii,_jj we calculate the position in the shared memory
										aux = Ims[ptr] - Nbh_s[(_kk * SM2) + (_iii * SM) + _jj];
										ptr++;
										// I do not know why CUDA does not like this line 
										//sumRet += pow(patch -patch_at,2);
										//sumRet += (patch - patch_at) *(patch - patch_at);
										sumRet += aux *aux;


									}// for patch
								}// for patch

							}// for patch

							 // div=2 * NN*beta*pow(sigma, 2) and has been calculate out of the kernel
							arg = sumRet / div;
							//aux = sumRet / (2 * NN*beta*pow(sigma, 2));
							weight = exp(-arg);
							// Group - Wise Label Propagation (accumulative part ) formula 2 in our paper (A=product/B=sum_weight)
							// Update A multiply weight by the label of the atlas for this voxel
							product += weight * lbl;
							// Update B
							sum_weight += weight;


							

						} // end for _jj
					}// end for _iii
				}// end for _kk
			}// end if Check that we can calculate the neighbourhood
			 // Wait until we have calculated the nh for one atlas
			__syncthreads();

		} // end for atlas
		  // Group - Wise Label Propagation (calculation part) formula 2 in our paper (A/B)
		  //if (blockIdx.x == 18 && blockIdx.y == 13 && blockIdx.z == 1){
		if (z >= half_window + half_patch && z < nz - half_window - half_patch &&
			x >= half_window + half_patch && x < nx - half_window - half_patch  &&
			y >= half_window + half_patch && y < ny - half_window - half_patch
			) {
			// if the product and sum_weight is greater than the precision
			if (abs(product) > DBL_EPSILON && abs(sum_weight) > DBL_EPSILON) {
				d_label_segmentation[ii] = (product / sum_weight);
				//d_label_segmentation[ii] = product;
			
			}
			else {
				// otherwise we asssign NAN and there is a process of regularization point 2.3 of our paper
				d_label_segmentation[ii] = NAN;
			}
		}

	} // if ii<n


} // end kernel



  // Input Image
float * im;
// Images in the atlas
float *atlas;
// Labels
float *label;
// Size of the image
int nx, ny, nz;
// Number of atlas and neighbourhood
int num, nh;
#ifdef MATLAB
// Functions
int WriteOutputMatFile(const char *file, const char *variable, const float *data, const int nx, const int ny, const int nz);
void ConvertMatrixMatlab(const float *matrix, float * matrix_matlab, const int nx, const int ny, const int nz);
void AnalyzeStructArray(const mxArray *sPtr, const char *fName, float *matrix, const int item, const int items);
void ConvertToRowBasedII(const mxArray *pa, float *matrix, const int item, const int items);
float* ConvertToRowBased(const mxArray *pa);
int Choice(const char * name);
bool FindDimension(const char *file);
bool CheckDimension(const mxArray *sPtr, const char *fName, int & dimX, int & dimY, int & dimZ);
#endif
// Regularization step
double Median(const float *patchImg, const int length);
void RegularizationStep(const int half_window, const int half_patch, const int nx, const int ny, const int nz, float *segmentationArray);
void RoundMatrix(float * mat, const int size);

int ReadMatFile(const char *file);

void Query();

///////////////////

// ********************** MatrixIO ******************************************//
/* Simpe class to read and store matrices from plain text to T* */
template<typename T>
class MatrixIO
{

public:
	MatrixIO();
	~MatrixIO();
	void Read(std::string file, T* vect);
	void Read(std::string file, T*& vect, vector<int>& dimensions);
	void Store(std::string fileName, const T* vect, const int num, const int rows, const int cols, const int depth);

};


template<typename T>
MatrixIO<T>::MatrixIO() {
}

template<typename T>
MatrixIO<T>::~MatrixIO() {
}

/** Read a matrix from a file knowing the dimensions
*  @param file: name of the file where the matrix is
*  @param vect: vector to fill with the data from the file
*/
template <typename T>
void MatrixIO<T>::Read(std::string file, T* vect) {
	std::string line;
	std::ifstream infile(file);
	std::string rowsStr, colsStr;
	std::getline(infile, rowsStr);
	std::getline(infile, colsStr);
	int index = 0;
	int rows = atoi(rowsStr.c_str());
	int cols = atoi(colsStr.c_str());
	int size = cols *rows;
	while (std::getline(infile, line))  // this does the checking!
	{
		std::istringstream iss(line);
		T c;

		while (iss >> c)
		{


			if (index < size) {
				vect[index] = c;
			}
			else {
				std::cout << "Error" << std::endl;
			}
			index++;
		}
	}
}
/** Read a matrix from a file and fill out the dimensions. it allocates vect with dimensions
*  @param file: name of the file where the matrix is
*  @param vect: vector to fill with the data from the file
*  @param dimensions: dimensions of the matrix
*/
template <typename T>
void MatrixIO<T>::Read(std::string file, T*& vect, vector<int>& dimensions) {
	std::string line;
	std::ifstream infile(file);
	std::string numStr, rowsStr, colsStr, depthStr;
	std::getline(infile, numStr);
	std::getline(infile, rowsStr);
	std::getline(infile, colsStr);
	std::getline(infile, depthStr);
	int index = 0;
	int num = atoi(numStr.c_str());
	int rows = atoi(rowsStr.c_str());
	int cols = atoi(colsStr.c_str());
	int depth = atoi(depthStr.c_str());
	int size = num *cols *rows *depth;
	if (size > 0) {
		dimensions.push_back(rows);
		dimensions.push_back(cols);
		dimensions.push_back(depth);
		dimensions.push_back(num);

		vect = (T*)malloc(sizeof(T)* size);
		while (std::getline(infile, line))  // this does the checking!
		{
			std::istringstream iss(line);
			T c;

			while (iss >> c)
			{


				if (index < size) {
					vect[index] = c;
				}
				else {
					std::cout << "Error" << std::endl;
				}
				index++;
			}
		}
	}
}

/** Store a matrix in a file
*  @param fileName: name of the variable in .mat file
*  @param vect: name of the variable in .mat file
*  @param num: number of matrices in the vect
*  @param rows: rows of the matrix
*  @param cols: cols of the matrix
*  @param depth: depth of the matrix
*/
template <typename T>
void MatrixIO<T>::Store(std::string fileName, const T* vect, const int num, const int rows, const int cols, const int depth) {
	std::ofstream myfile(fileName);
	if (myfile.is_open())
	{
		int size = num* rows*cols*depth;
		myfile << num << "\n";
		myfile << rows << "\n";
		myfile << cols << "\n";
		myfile << depth << "\n";
		for (int count = 0; count < size; count++) {
			//std::cout << vect[count] << std::endl;
			myfile << vect[count] << " ";
		}
		myfile.close();
	}
	else std::cout << "Unable to open file";

}


// ********************** MatrixIO ******************************************//





#ifdef MATLAB
/** Calculate the median of the array
*  @param name: name of the variable in .mat file
*/
int Choice(const char * name) {
	int option = -1;
#ifdef DEBUG_ALG 
	printf("NAME %s\n", name);
#endif
	if (strcmp(name, "img") == 0) {
		option = 0;
	}
	else {
		if (strcmp(name, "atlas") == 0) {
			option = 1;
		}
		else {
			if (strcmp(name, "atlas_label") == 0) {
				option = 2;
			}

		}

	}

	return option;

}




/** Calculate the median of the array
*  @param pa: name of the variable in .mat file
*/
float* ConvertToRowBased(const mxArray *pa) {
	int r, c, z;
	float * realPart = (float *)mxGetData(pa);
	float *matrix;
	size_t size_array = mxGetNumberOfElements(pa);
	matrix = (float *)malloc(sizeof(float)*size_array);

	const  mwSize *size = mxGetDimensions(pa);
	for (int ii = 0; ii<mxGetNumberOfDimensions(pa); ii++) {
#ifdef DEBUG_ALG 
		std::cout << "convertToRowBased " << size[ii] << std::endl;
#endif
	}
	if (mxGetNumberOfDimensions(pa) == 3) {
		r = size[0];
		c = size[1];
		z = size[2];

	}
	nx = r;
	ny = c;
	nz = z;

	int i = 0;

	for (int zI = 0; zI<z; zI++) {
		//printf("zI=%d\n",zI);
		for (int rI = 0; rI<r; rI++) {
			for (int cI = 0; cI<c*r; cI += r) {
				matrix[i] = realPart[(zI *r*c) + rI + cI];
				i++;
			}
		}

	}
	return matrix;

}
/** Convert from column-major to row-major
*  @param pa: name of the variable in .mat file
*  @param matrix: data
*  @param item: specified element in the anatomy atlas
*  @param items: number of elements of the array
*/
void ConvertToRowBasedII(const mxArray *pa, float *matrix, const int item, const int items) {
	int r, c, z;
	float * realPart = (float *)mxGetData(pa);

	size_t size_array = mxGetNumberOfElements(pa);


	const  mwSize *size = mxGetDimensions(pa);

	if (mxGetNumberOfDimensions(pa) == 3) {
		r = size[0];
		c = size[1];
		z = size[2];

	}
	nx = r;
	ny = c;
	nz = z;
#ifdef DEBUG_ALG 
	printf("size_array %d r=%d c=%d z=%d %d\n", size_array, r, c, z, size_array == r*c*z);
	printf("size_array %d nx=%d ny=%d nz=%d %d\n", size_array, nx, ny, nz, size_array == nx*ny*nz);

#endif


	int i = 0;
	int offset = item * nx*ny*nz;

	for (int zI = 0; zI<z; zI++) {

		for (int rI = 0; rI<r; rI++) {
			for (int cI = 0; cI<c*r; cI += r) {
				matrix[i + offset] = realPart[(zI *r*c) + rI + cI];


				i++;
			}
		}

	}
}

/**  Analyze field FNAME in struct array SPTR
*  C:\Program Files\MATLAB\R2018b\extern\examples\eng_mat
*  @param sPtr: name of the variable in .mat file
*  @param fName: data
*  @param matrix: data
*  @param item: specified element in the anatomy atlas
*  @param items: number of elements of the array
*/


void
AnalyzeStructArray(const mxArray *sPtr, const char *fName, float *matrix, const int item, int items)
{
	mwSize nElements;       /* number of elements in array */
	mwIndex eIdx;           /* element index */
	const mxArray *fPtr;    /* field pointer */
	float *realPtr;        /* pointer to data */
	float total;           /* value to calculate */

	total = 0;
	nElements = (mwSize)mxGetNumberOfElements(sPtr);
	for (eIdx = 0; eIdx < nElements; eIdx++) {
		fPtr = mxGetField(sPtr, eIdx, fName);
		if ((fPtr != NULL)
			&& (mxGetClassID(fPtr) == mxSINGLE_CLASS)
			&& (!mxIsComplex(fPtr)))
		{
			realPtr = (float*)mxGetData(fPtr);
			total = total + realPtr[0];
			ConvertToRowBasedII(fPtr, matrix, item, items);
		}
	}
	//printf("Total for %s: %.2f\n", fName, total);
}

/**  check whether the dimension in sPtr variable match in the .mat file
*  C:\Program Files\MATLAB\R2018b\extern\examples\eng_mat
*  @param sPtr: name of the variable in .mat file
*  @param fName: data
*  @param dimX: dimension in x axis
*  @param dimY: dimension in y axis
*  @param dimZ: dimension in z axis
*/

bool CheckDimension(const mxArray *sPtr, const char *fName, int & dimX, int & dimY, int & dimZ) {
	mwSize nElements;       /* number of elements in array */
	mwIndex eIdx;           /* element index */
	const mxArray *fPtr;    /* field pointer */
	float *realPtr;        /* pointer to data */
	float total;           /* value to calculate */
	int r, c, z;
	total = 0;
	bool valid = true;
	nElements = (mwSize)mxGetNumberOfElements(sPtr);
	for (eIdx = 0; eIdx < nElements && valid; eIdx++) {
		fPtr = mxGetField(sPtr, eIdx, fName);
		if ((fPtr != NULL)
			&& (mxGetClassID(fPtr) == mxSINGLE_CLASS)
			&& (!mxIsComplex(fPtr)))
		{

			const  mwSize *size = mxGetDimensions(fPtr);

			if (mxGetNumberOfDimensions(fPtr) == 3) {
				r = size[0];
				c = size[1];
				z = size[2];
#ifdef DEBUG_ALG 
				std::cout << "eIdx " << eIdx << " " << r << " c " << c << " z " << z << std::endl;
#endif

				if (dimX == -1 && dimY == -1 && dimZ == -1) {
					dimX = r;
					dimY = c;
					dimZ = z;
				}
				else {
					if (dimX != r && dimY != c && dimZ != z) {
						std::cout << "Dimensions do NOT match in struct " << fName << std::endl;
						valid = false;
					}
				}
			}

		}
	}
	return valid;

}


/**
* Read the anatomy data
* 271x271x221 is the dimension to ilustrate
*  struct with fields for the anatomy:
*		- atlas: [1×1 struct]
*			- A.atlas struct with fields:
MRI1: [271x271x221 single]
MRI2: [271x271x221 single]
...
MRIn: [271x271x221 single]
*		- atlas_label: [1×1 struct]
MRI1: [271x271x221 single]
MRI2: [271x271x221 single]
...
MRIn: [271x271x221 single]
*		- img: [271x271x221  single]
*/
int ReadMatFile(const char *file) {
	MATFile *pmat;
	const char **dir;
	const char *name;
	int      ndir;
	int      i;
	mxArray *pa;
#ifdef DEBUG_ALG 
	std::cout << "Reading file " << file << std::endl;
#endif

	/*
	* Open file to get directory
	*/
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		std::cout << "Error opening file " << file << std::endl;
		return(1);
	}

	/*
	* get directory of MAT-file
	*/
	dir = (const char **)matGetDir(pmat, &ndir);
	if (dir == NULL) {
		std::cout << "Error reading directory of file  " << file << std::endl;
		return(1);
	}

	mxFree(dir);

	/* In order to use matGetNextXXX correctly, reopen file to read in headers. */
	if (matClose(pmat) != 0) {
		std::cout << "Error closing file " << file << std::endl;
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		std::cout << "Error reopening file " << file << std::endl;
		return(1);
	}

	/* Get headers of all variables */
	// printf("\nExamining the header for each variable:\n");
	for (i = 0; i < ndir; i++) {
		pa = matGetNextVariableInfo(pmat, &name);
		if (pa == NULL) {
			std::cout << "Error reading in file " << file << std::endl;
			return(1);
		}

		mxDestroyArray(pa);
	}

	/* Reopen file to read in actual arrays. */
	if (matClose(pmat) != 0) {
		std::cout << "Error closing file " << file << std::endl;
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		std::cout << "Error reopening file" << file << std::endl;
		return(1);
	}

	/* Read in each array. */

	for (i = 0; i<ndir; i++) {
		pa = matGetNextVariable(pmat, &name);
		if (pa == NULL) {
			std::cout << "Error reading in file" << file << std::endl;
			return(1);
		}
		void * p = mxGetData(pa);
		/*
		* Diagnose array pa
		*/




		if (mxIsStruct(pa)) {
#ifdef DEBUG_ALG 
			std::cout << "Struct " << mxGetNumberOfFields(pa) << std::endl;
#endif


			int option = Choice(name);
			switch (option) {

			case 1: //atlas

				atlas = (float *)malloc(sizeof(float) * nx * ny * nz * num);

				for (int ii = 0; ii<mxGetNumberOfFields(pa); ii++) {
					AnalyzeStructArray(pa, mxGetFieldNameByNumber(pa, ii), atlas, ii, num);
				}
				break; //label
			case 2:
				label = (float *)malloc(sizeof(float) * nx * ny * nz * num);

				for (int ii = 0; ii<mxGetNumberOfFields(pa); ii++) {

					AnalyzeStructArray(pa, mxGetFieldNameByNumber(pa, ii), label, ii, num);
				}
				break;
			}
		}
		else {
			im = ConvertToRowBased(pa);
		}
		mxDestroyArray(pa);
	}

	if (matClose(pmat) != 0) {
		std::cout << "Error closing file " << file << std::endl;
		return(1);
	}

	std::cout << "Read Completed " << std::endl;
	return(0);
}


/**  Get the dimension for the variables in the anatomy atlas in the .mat file
*  C:\Program Files\MATLAB\R2018b\extern\examples\eng_mat
*  @param file: name of the .mat file
*/
bool FindDimension(const char *file) {
	MATFile *pmat;
	const char **dir;
	const char *name;
	int      ndir;
	int      i;
	mxArray *pa;
#ifdef DEBUG_ALG 
	std::cout << "Reading file " << file << std::endl;
#endif
	bool valid = true;
	/*
	* Open file to get directory
	*/
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		std::cout << "Error opening file " << file << std::endl;
		return(1);
	}

	/*
	* get directory of MAT-file
	*/
	dir = (const char **)matGetDir(pmat, &ndir);
#ifdef DEBUG_ALG 
	std::cout << "matGetDir  " << dir << " ndir " << ndir << std::endl;
#endif

	if (dir == NULL) {
		std::cout << "Error reading directory of file  " << file << std::endl;
		return(1);
	}

	mxFree(dir);

	///* In order to use matGetNextXXX correctly, reopen file to read in headers. */
	if (matClose(pmat) != 0) {
		std::cout << "Error closing file " << file << std::endl;
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		std::cout << "Error reopening file " << file << std::endl;
		return(1);
	}
	std::string variables[] = { "atlas","atlas_label","img" };
	/* Get headers of all variables */
	// printf("\nExamining the header for each variable:\n");
	//int dimX = -1,dimY=-1,dimZ=-1;
	nx = -1; ny = -1; nz = -1;
	num = -1;
	int numAux;

	for (i = 0; i < ndir && valid; i++) {
		pa = matGetNextVariableInfo(pmat, &name);
#ifdef DEBUG_ALG 
		std::cout << "Name " << name << " Num " << num << std::endl;
		std::cout << "variables " << strcmp(name, variables[i].c_str()) << std::endl;
#endif

		if (mxIsStruct(pa)) {
			numAux = mxGetNumberOfFields(pa);
			if (num == -1) {
				num = numAux;
			}
			else {
				if (num != numAux) {
					std::cout << "Num does not match" << std::endl;
				}
			}
			for (int ii = 0; ii<num; ii++) {
#ifdef DEBUG_ALG 
				std::cout << "dimensions " << nx << " " << ny << " " << nz << std::endl;
#endif

				valid = valid && CheckDimension(pa, mxGetFieldNameByNumber(pa, ii), nx, ny, nz);
			}
		}
		else {


			const  mwSize *size = mxGetDimensions(pa);

			if (mxGetNumberOfDimensions(pa) == 3) {
				if (nx != size[0] || ny != size[1] || nz != size[2]) {
					std::cout << "Name " << name << " do NOT match the dimension " << std::endl;
					std::cout << "dimensions " << nx << " " << ny << " " << nz << std::endl;
					std::cout << "found " << size[0] << " " << size[1] << " " << size[2] << std::endl;
					valid = valid && false;
				}


			}


		}

		if (pa == NULL) {
			std::cout << "Error reading in file " << file << std::endl;
			return(1);
		}

		mxDestroyArray(pa);
	}


	/* Reopen file to read in actual arrays. */
	if (matClose(pmat) != 0) {
		std::cout << "Error closing file " << file << std::endl;
		return(1);
	}


	std::cout << "Find Dimensions Completed " << std::endl;
	return valid;
}
/**  Convert from row-major to column-major order
*  @param matrix: matrix in matlab row-major order
*  @param matrix_matlab: matrix in matlab column-major order
*  @param nx: dimension in x axis
*  @param ny: dimension in y axis
*  @param nz: dimension in z axis
*/
void ConvertMatrixMatlab(const float *matrix, float * matrix_matlab, const int nx, const int ny, const int nz) {
	int i = 0;
	for (int z = 0; z<nz; z++) {
		for (int y = 0; y<ny; y++) {
			for (int x = 0; x<nx*ny; x += ny) {
				matrix_matlab[i] = matrix[z*(nx*ny) + y + x];
				i++;
			}
		}
	}
}

/**  Write the data onto the .mat file
*  C:\Program Files\MATLAB\R2018b\extern\examples\eng_mat
*  @param file: name of the Matlab file
*  @param variable: name of the variable in Matlab
*  @param data: data to store
*  @param nx: dimension in x axis
*  @param ny: dimension in y axis
*  @param nz: dimension in z axis
*/
int WriteOutputMatFile(const char *file, const char *variable, const float *data, const int nx, const int ny, const int nz) {
	MATFile *pmat;
	mxArray *pa1;
	float *data_matlab;
	size_t dims[3] = { nx, ny, nz };
	int size = dims[0] * dims[1] * dims[2];


	data_matlab = (float *)malloc(sizeof(float) * size);
	int status;
	ConvertMatrixMatlab(data, data_matlab, dims[0], dims[1], dims[2]);
#ifdef DEBUG_ALG 
	std::cout << "Creating file " << file << " " << data[0] << std::endl;
#endif
	pmat = matOpen(file, "w");
	if (pmat == NULL) {
		std::cout << "Error creating file " << file << std::endl;
		std::cout << "Do you have write permission in this directory? " << file << std::endl;

		return(EXIT_FAILURE);
	}

	pa1 = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
	if (pa1 == NULL) {
		std::cout << __FILE__ << ": Out of memory on line " << __LINE__ << std::endl;
		std::cout << "Unable to create mxArray " << file << std::endl;
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(pa1)), (void *)data_matlab, size * sizeof(float));
	status = matPutVariable(pmat, variable, pa1);
	if (status != 0) {
		std::cout << __FILE__ << " :  Error using matPutVariable on line " << __LINE__ << std::endl;
		return(EXIT_FAILURE);
	}

	/* clean up */
	mxDestroyArray(pa1);


	if (matClose(pmat) != 0) {
		std::cout << "Error closing file" << file << std::endl;
		return(EXIT_FAILURE);
	}


	std::cout << "File " << file << " written" << std::endl;
	return(EXIT_SUCCESS);
}

#else

bool FindDimension(char *file) {
	return true;
}

// http://www.cplusplus.com/doc/tutorial/files/
int ReadMatFile(const char *file) {
	// Read the configuration from the file and create the random matrix
	int randomData;
	string line;
	string atlasStr, labelStr, imStr;
	ifstream myfile(file);
	int choice = 0;
	if (myfile.is_open())
	{
		/* initialize random seed: */
		srand(time(NULL));
		/* generate secret number between 1 and 10: */
		std::cout << "Number " << (float)rand() / (float)RAND_MAX;
		while (getline(myfile, line))
		{
			cout << line << '\n';
			switch (choice) {
			case 0:
				randomData = atoi(line.c_str());
				choice++;
				break;
			case 1:
				if (randomData)
					num = atoi(line.c_str());
				else
					atlasStr = line;
				choice++;
				break;
			case 2:

				if (randomData)
					nx = atoi(line.c_str());
				else
					labelStr = line;
				choice++;
				break;
			case 3:

				if (randomData)
					ny = atoi(line.c_str());
				else
					imStr = line;
				choice++;
				break;
			case 4:
				if (randomData) {
					nz = atoi(line.c_str());
					choice++;
				}
				break;
			}

		}
		if (randomData) {
			std::cout << "Creating random matrices "<< num << std::endl;
		
			int elements = nx * ny * nz * num;
			std::cout << "Size " << elements << std::endl;
			atlas = (float*)malloc(sizeof(float)* nx * ny * nz * num);
			label = (float*)malloc(sizeof(float)* nx * ny * nz * num);
			im = (float*)malloc(sizeof(float)* nx * ny * nz);
			// Atlas label
			for (int i = 0; i < elements; i++) {
				atlas[i] = 10.0f* ((float)rand() / (float)RAND_MAX);
				label[i] = 10.0f*((float)rand() / (float)RAND_MAX);
			}
			for (int i = 0; i < nx * ny * nz; i++) {
				im[i] = 10.0f*((float)rand() / (float)RAND_MAX);

			}
			/*MatrixIO<float> matrixIO;
			matrixIO.Store("atlas.txt", atlas, num, nx, ny, nz);
			matrixIO.Store("label.txt", label, num, nx, ny, nz);
			matrixIO.Store("im.txt", im, 1, nx, ny, nz);*/
		}
		else {
			std::cout << "Reading data" << std::endl;
			MatrixIO<float> matrixIO;
			vector<int> dimensionsAtlas;
			vector<int> dimensionsLabel;
			vector<int> dimensionsIm;

			matrixIO.Read(atlasStr, atlas, dimensionsAtlas);
			matrixIO.Read(labelStr, label, dimensionsLabel);
			matrixIO.Read(imStr, im, dimensionsIm);

			if (dimensionsAtlas[0] == dimensionsLabel[0] &&
				dimensionsAtlas[1] == dimensionsLabel[1] &&
				dimensionsAtlas[2] == dimensionsLabel[2] &&
				dimensionsAtlas[3] == dimensionsLabel[3] &&
				dimensionsAtlas[0] == dimensionsIm[0] &&
				dimensionsAtlas[1] == dimensionsIm[1] &&
				dimensionsAtlas[2] == dimensionsIm[2]
				) {

				nx = dimensionsLabel[0];
				ny = dimensionsLabel[1];
				nz = dimensionsLabel[2];
				num = dimensionsLabel[3];
			}
			else {
				std::cout << "Dimensions do not match" << std::endl;
			}

		}

		myfile.close();
	}

	else cout << "Unable to open file";

	return 0;
}
#endif







/** Calculate the median of the array
*  @param patchImg: patch array to calculate the median
*  @param length: size of the array
*/
void RoundMatrix(float * mat, const int size) {
	for (int i = 0; i < size; i++) {
		mat[i] = round(mat[i]);
	}
}


/** Calculate the median of the array
*  @param patchImg: patch array to calculate the median
*  @param length: size of the array
*/
double Median(const float *patchImg,const int length) {

	float vector[27];

	int j;
	int cnt = 0;
	for (j = 0; j < length; j++) {
		if (!isnan(patchImg[j])) {
			vector[cnt] = patchImg[j];
			cnt++;
		}
	}
	if (cnt <= 0) std::cout << "#############PANIC######### " << std::endl;
	std::stable_sort(vector, vector + cnt);

	if (cnt % 2 == 0) {
		double a = vector[cnt / 2 - 1];
		double b = vector[cnt / 2];
		return (a + b) / 2;
	}
	else {
		return (double)vector[cnt / 2];
	}
}

/** Remove NaN with hte median of the neighbourhood see formula (3)  [1]
*  @param half_window: half of the patch window
*  @param half_patch: half of the patch dimension
*  @param nx: dimension x
*  @param ny: dimension ny
*  @param nz: dimension z
*  @param segmentationArray: array to remove the NaN
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017).
*
*/
void RegularizationStep(const int half_window, const int half_patch, const int nx, const int ny, const int nz, float *segmentationArray) {
	int level1 = nx*ny;
	float patchImg[27];
	bool nanFlag = true;
	int numLops = 0;
	int numNaN = 0;
	int numMax = 0;

	while (nanFlag) {
		nanFlag = false;
		numLops++;
		for (int z = half_window + half_patch; z < nz - half_window - half_patch; z++) {
			for (int x = half_window + half_patch; x < nx - half_window - half_patch; x++) {
				for (int y = half_window + half_patch; y < ny - half_window - half_patch; y++) {
					
					if (isnan(segmentationArray[z *level1 + x * ny + y])) {
				
						// Extract a patch in the image
						numNaN++;
						int cnt = 0;
						for (int r = z - half_patch; r <= z + half_patch; r++) {
							for (int s = x - half_patch; s <= x + half_patch; s++) {
								for (int t = y - half_patch; t <= y + half_patch; t++) {
									patchImg[cnt] = segmentationArray[r * level1 + s * ny + t];
									cnt++;
								}
							}
						}
						// Compute the median
						segmentationArray[z *level1 + x *ny + y] = Median(patchImg, 27);
						if (isnan(segmentationArray[z *level1 + x *ny + y])) {
							nanFlag = true;
						}
						else {
							if (segmentationArray[z *level1 + x *ny + y] == 0) {
								numMax++;
							}
						}
						//std::cout << "Index" << z *level1 + x * ny + y << " Value " << segmentationArray[z *level1 + x *ny + y] << std::endl;
					}

				}
			}
		}
	}
	float numNanFlt = numNaN;
	float total = nx *ny *nz;
	std::cout << "NumLops " << numLops << " NumNaN " << numNaN << " % " << 100.0f *(numNanFlt / total) << " NumMax " << numMax << std::endl;
}
/** Calculate the numbers needed for shared memory to load the elements in the array 
*  @param nh: neighbourd size
*  @param patch: patch size
*  @param block: block size
*  @param LIMIT: number of threads before change the set. 
*  @param SH: size of the shared memory only one side => SH^3
*  @param LOAD1: number of elements to load for first set of threads
*  @param LOAD2: number of elements to load for second set of threads
* Example : block 10 patch 3 nh 7
* // int LIMIT 168, int LOAD1 5, int LOAD2 6, int SM 18, int SM2  18 * 18
* 
*/
void calculateLoadThreads(const int nh, const int patch, const int  block, int & LIMIT,int &SH, int &LOAD1, int &LOAD2) {
	
	SH = block + 2 * (nh / 2) + 2 * (patch / 2);
	int sh_size = SH *SH *SH;
	int block_size = block *block *block;
	LOAD1 = sh_size / block_size;
	LOAD2 = LOAD1 + 1;
	
	LIMIT = LOAD2* block_size - sh_size;

	std::cout << "LIMIT " << LIMIT << " SH " << SH << " LOAD1 " << LOAD1 << " LOAD2 "<< LOAD2  << std::endl;
}





/** Synthesise a new modality from an input image using an anatomy atlas. An anatomy atlas is a pair of aligned volumes {MRI,CT}
*  @param img: input image for example MRI volume
*  @param atlas: atlas images for example MRI
*  @param atlas_label: labels for the modality to build in the atlas
*  @param choice: execution choice {GM, GM2, SM}
*  @param label_segmentation: synthetic image for the new modality
*  @param div: see formula (1) 2S beta sigma^2 in [1]
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017).
https://doi.org/10.1007/s11554-016-0568-0
*/
void MultiPatchBasedSegmentationGpu(const float * img,const float *atlas,const float *atlas_label, float * label_segmentation, EXEC_CHOICE choice, const double div) {

	float * d_img, *d_atlas, *d_atlas_label;
											
	float *d_label_segmentation=NULL, *d_label_segmentationPrev=NULL;
	int elements = nx*ny*nz;
	int elementsII = nx*ny*nz*num;
	size_t size = elements * sizeof(float);
	size_t sizeII = elementsII * sizeof(float);
	std::cout << "MultiPatchBasedSegmentationGpu" << std::endl;
	
	int patch_size = 3;
	float NN = static_cast<float>(patch_size *patch_size*patch_size);
	
	int window = nh;
	int half_patch = (patch_size - 1) / 2;
	int half_window = (window - 1) / 2;

	int  LIMIT, SH, LOAD1, LOAD2;

	int block_size_x;
	int block_size_y;
	int block_size_z;
	// Have a look at Bloques de Threads en Patch-Based from Gmail
	if (choice == 0 || choice == 1) {
	
		block_size_x = 8;
		block_size_y = 8;
		block_size_z = 10;
	}
	else {

		block_size_x = 10;
		block_size_y = 10;
		block_size_z = 10;
	}

	
	int gridX = (nx / block_size_x) + (nx % block_size_x == 0 ? 0 : 1);
	int gridY = (ny / block_size_y) + (ny % block_size_y == 0 ? 0 : 1);
	int gridZ = (nz / block_size_z) + (nz % block_size_z == 0 ? 0 : 1);
	

	double alpha = 0;
	
	
	dim3 threadsBlock(block_size_x, block_size_y, block_size_z);
	dim3 grid(gridX, gridY, gridZ);

	std::cout << "patch_size" << patch_size << " half_patch " << half_patch << " half_window=" << half_window << std::endl;


	checkCudaErrors(cudaMalloc(&d_img, size));

	checkCudaErrors(cudaMalloc(&d_atlas, sizeII));
	checkCudaErrors(cudaMalloc(&d_atlas_label, sizeII));
	
	checkCudaErrors(cudaMalloc(&d_label_segmentation, size));
	checkCudaErrors(cudaMemset(d_label_segmentation, 0, size));

	
	// Copy matrices from the host to device
	checkCudaErrors(cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_atlas, atlas, sizeII, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_atlas_label, atlas_label, sizeII, cudaMemcpyHostToDevice));

	
	// Compute
	
	int sharedMemory;
	
	std::cout << "Grid [" << gridX << "," << gridY << "," << gridZ << "]" << std::endl;
	std::cout << "Block [" << block_size_x << "," << block_size_y << "," << block_size_z << "]" << std::endl;
	
	
	switch (choice) {
	case EXEC_CHOICE::GM:
		std::cout << "CUDA-GM"<< std::endl;
		MultiPatchSegmentationGMKernel << <grid, threadsBlock >> >(d_img, d_atlas, d_atlas_label, d_label_segmentation, elements, NN, nx, ny, nz, div, patch_size, window, half_patch, half_window, num);
		break;
	case EXEC_CHOICE::GM2:
		std::cout << "CUDA-GM2" << std::endl;
		MultiPatchSegmentationGM2Kernel << <grid, threadsBlock >> >(NULL, d_img, d_atlas, d_atlas_label, d_label_segmentation, elements, div, nx, ny, nz, patch_size, window, half_patch, half_window, num);
		break;
	case EXEC_CHOICE::SM:
		calculateLoadThreads(nh, patch_size, block_size_x, LIMIT, SH, LOAD1, LOAD2);
		sharedMemory = SH * SH * SH * sizeof(float);
		MultiPatchSegmentationSMKernel << <grid, threadsBlock, sharedMemory >> >(d_img, d_atlas, d_atlas_label, d_label_segmentation, elements, div, nx, ny, nz, patch_size, half_patch, half_window, num, nh, LIMIT, LOAD1, LOAD2, SH, SH  *SH);
	

		std::cout << "CUDA-SM " << sharedMemory << " bytes"<<std::endl;
		
													
		
		
		break;	

	

	}
	// To check the errors you need cudaDeviceSync
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error !=cudaSuccess )
		std::cout << "Error " <<  cudaGetErrorString(error);

	
	checkCudaErrors(cudaMemcpy(label_segmentation, d_label_segmentation, size, cudaMemcpyDeviceToHost));
	
	CLEANUP("Clean UP PatchBasedSegmentationGpu");

	
	std::cout << "EnD" << std::endl;


}



void print(const char* str, float * matrix, const int elements) {
	printf("%s\n", str);
	for (int i = 0; i < elements; i++) {
		//printf("%f\n", matrix[i]);
		printf("ptr %d Value %f\n", i, matrix[i]);
	}
}

void PrintBlock(const char* str, block * matrix, const int elements, const int nx, const int ny) {
	printf("%s\n", str);
	for (int i = 0; i < elements; i++) {

		//int z = i / (nx*ny);
		//int aux = (i - (z*nx*ny));
		//// ROw
		//int x = aux / ny;
		//// Col
		//int y = aux% ny;

		//int bx = x / 10;
		//int by = y / 10;
		//int bz = z / 10;


		//int thx = x % 10;
		//int thy = y % 10;
		//int thz = z % 10;
		//if (matrix[i].bx==0&& matrix[i].by==0 && matrix[i].bz==0)
		//printf("i %d index %f BLK [%d,%d,%d] TH [%d,%d,%d] COR [%d,%d,%d] _BLK [%d,%d,%d]  _TH [%d,%d,%d] \n", i, matrix[i].index, matrix[i].bx, matrix[i].by, matrix[i].bz, matrix[i].thx, matrix[i].thy, matrix[i].thz, x, y, z, bx, by, bz, thx, thy, thz);
		//printf("i %d val %f ii %d BLK [%d,%d,%d] REF [%d,%d,%d] XYZ [%d,%d,%d] IND [%d,%d,%d] TH [%d,%d,%d]  \n", i, matrix[i].index, matrix[i].ii, matrix[i].bx, matrix[i].by, matrix[i].bz, matrix[i].refX, matrix[i].refY, matrix[i].refZ, matrix[i].x, matrix[i].y, matrix[i].z, matrix[i].xIndex, matrix[i].yIndex, matrix[i].zIndex, matrix[i].thx, matrix[i].thy, matrix[i].thz);
		//printf("at %d ptr %d XYZ [%d,%d,%d] TH [%d,%d,%d]  Value %f\n", i / 10648, i, matrix[i].xIndex, matrix[i].yIndex, matrix[i].zIndex, matrix[i].thx, matrix[i].thy, matrix[i].thz, matrix[i].index);
		// Out
		//printf("ii %d at %d ptr %d XYZ [%d,%d,%d] TH [%d,%d,%d]  Value %f\n",matrix[i].ii, i / 10648, i, matrix[i].xIndex, matrix[i].yIndex, matrix[i].zIndex, matrix[i].thx, matrix[i].thy, matrix[i].thz, matrix[i].index);
		// Out 2
		//printf("AT %d [%d-%d] [%d-%d] [%d-%d] [%d-%d] [%d-%d] [%d-%d] TH [%d,%d,%d] Value %f Product %f Sum %f\n", matrix[i].ii, matrix[i].refX, matrix[i].x, matrix[i].refY, matrix[i].y, matrix[i].refZ, matrix[i].z, matrix[i].xIndex, matrix[i].bx, matrix[i].yIndex, matrix[i].by, matrix[i].zIndex, matrix[i].bz, matrix[i].thx, matrix[i].thy, matrix[i].thz, matrix[i].index, matrix[i].product, matrix[i].sum);
		// Out 3
		//printf("AT %d BF [%d,%d,%d] GM [%d,%d,%d] Value %f\n", matrix[i].ii, matrix[i].refX, matrix[i].refY, matrix[i].refZ, matrix[i].x, matrix[i].y, matrix[i].z, matrix[i].index);
		//printf("AT %d BF [%d,%d,%d] GM [%d,%d,%d] Value %f\n", , matrix[i].refX, matrix[i].refY, matrix[i].refZ, matrix[i].x, matrix[i].y, matrix[i].z, matrix[i].index);
		//printf("AT %d Product %f Sum %f\n", matrix[i].ii, matrix[i].product, matrix[i].sum);
		printf("AT %d SumRet %f Arg %f Weight %.10e Product %.10e sum_weight %.10e lbl %f\n", matrix[i].ii, matrix[i].sumRet, matrix[i].arg, matrix[i].weight, matrix[i].product, matrix[i].sum_weight, matrix[i].lbl);
	}
}


/** Query device especifications
*/
void Query() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
	2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
}

/** Get the string of a choice
*
*  @param choice: choice to display
*  @return depending on the choice a string with SingleCore, MultiCore, SingleCoreTestTime, MultiCoreTestTime
*/
string GetChoice(const EXEC_CHOICE choice) {
	string choiceStr;
	switch (choice) {


	case EXEC_CHOICE::GM:
		choiceStr = "GM";
		break;
	case EXEC_CHOICE::GM2:
		choiceStr = "GM2";
		break;
	case EXEC_CHOICE::SM:
		choiceStr = "SM";
		break;

	default:
		choiceStr = "Unknown";
		break;
	}

	return choiceStr;
}




// Execution 
// PatchBasedPseudoCT data choice check name_output opt_patchbased_alg nh sigma delta 
// - data: .mat file with img (input MRI) atlas (CT) atlas_label (Labels)
// - choice 
//		* 0 CPU execution
//		* 2 Calculate CPU an espcific voxel (Debug purposes)
//		* 5 Calculate Shared memrory an espcific voxel (Debug purposes)
//		* 3 GPU execution going through different parameters nh opt_patchbased_alg atlas
//		* 4 GPU execution
// - check (NOT IN USED)
// - name_output mat file to write the pseudo-CT (mat files)
// - opt_patchbased_alg for GPU execution see https://link.springer.com/article/10.1007/s11554-016-0568-0 "Real-time patch-based medical image modality propagation by GPU computing"
//		* 0 CUDA-GM Gpu with no optimization
//		* 1 CUDA-GM2
//		* 2 CUDA-SM nH 11x11x11 6 CUDA-SM nH 9x9x9 7 CUDA-SM nH 7x7x7
// - nh neighbourd possible values: 7 9 11 
// - sigma
// - delta
// Example
// New compilation PatchBasedPseudoCT D:\Users\ealcain\DB_Images\JRTIP\02_CT_Propagation\testRandom.mat 4 0 testRandom_new 7 7 2.5 1 > testRandom_new.txt
// Old compilation D:\Users\ealcain\Documents\Visual Studio 2013\Projects\NewPatchedBrainOp\x64\Release
// NewPatchedBrainOp  D:\Users\ealcain\DB_Images\JRTIP\02_CT_Propagation\testRandom.mat 4 0 testRandom 7 7 2.5 1 > testRandom.txt

// How to check the result in Matlab
//  slice = 87;
// showResult(slice);

// PatchBasedPseudoCT Data.mat EXEC_CHOICE NH [TH] [sigma] [beta]
//	- [1] Data.mat Anatomy atlas I, \mathcal{A} =\{(\mathcal{I}^{i},L^{i})\}^n_{i=1}. Atlas dimension is inferred from the data
//  - [2] Name for the mat file result
//	- [3] EXEC_CHOICE {GM = 0, 	GM2 = 1, 	SM = 2, 	GpuTestTime=4 } 
//  - [4] NH {7,9,11}
//	- [5] sigma 1 default (2.5) optional only with MultiCore/SingleCore. sigma is the standard deviation of the noise in the images given by Signal Noise Ratio (SNR), but we can expect that images have a good SNR.
//	- [6] beta 1 default optional only with MultiCore/SingleCore. Beta is a positive real number that influences the difficulty to accept patches with less or more similarity
//	- [7] cudaDevice 0 default optional only with MultiCore/SingleCore. Beta is a positive real number that influences the difficulty to accept patches with less or more similarity


//	
// Examples

// GM  Nh=7 
// PatchBasedPseudoCT data\Subject_08_AtlasSize_01.mat Subject_08_AtlasSize_01_7Nh_MC_TH12 0 7 
// GM2  Nh=7 sigma=2.5 beta=1.0
// PatchBasedPseudoCT data\Subject_08_AtlasSize_01.mat Subject_08_AtlasSize_01_7Nh_MC_TH12 1 7 2.5 1
// SM  Nh=7 sigma=2.5 beta=1.0
// PatchBasedPseudoCT data\Subject_08_AtlasSize_01.mat Subject_08_AtlasSize_01_7Nh_MC_TH12 2 7 2.5 1
// GpuTestTime 
// PatchBasedPseudoCT  data\\Subject_08_AtlasSize_18.mat Subject_01_AtlasSize_01_Op2_Rd_7Nh 4 7

int main(int argc, char **argv)
{
	int choice_op;
	double rB;
	double wB;
	double div;
	double bw_eff;
	std::map<int, std::map<int, std::vector<float>>> resultsMap;
	std::map<int, std::map<int, std::vector<float>>>::iterator it;
	std::map<int, std::vector<float>>::iterator it2;
	string partial_total_results_path = "D:\\Users\\ealcain\\partialTotalresultsSM_9_0.txt";
	string partial_results_path = "D:\\Users\\ealcain\\partialresultsSM_9_0.txt";
	string final_results_path = "D:\\Users\\ealcain\\finalresultsSM_9_0.txt";
	
	int result;
	// http://www.cplusplus.com/reference/ctime/localtime/
	time_t rawtime;
	struct tm * timeinfo;
	Query();
#ifdef WINDOWS
	LPSYSTEMTIME lpSystemTime;

	lpSystemTime = (LPSYSTEMTIME)malloc(sizeof(SYSTEMTIME));
	GetLocalTime(lpSystemTime);
	std::cout << "Local Time Starts " << lpSystemTime->wHour << ":" << lpSystemTime->wMinute << ":" << lpSystemTime->wSecond << std:endl;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Local Time Starts: %s", asctime(timeinfo));
#else
	//
#endif
	
	
	if (argc >=4) {
		bool checkValidData = FindDimension(argv[1]);
		if (checkValidData) {
			std::cout << "nx=" << nx << " ny=" << ny << " nz=" << nz << " num=" << num << std::endl;

			result = ReadMatFile(argv[1]);
			int num_threads, elementsAnatomyAtlas, elements;

			string anatomyAtlas;
			string resultName;
			
			bool writeResult = true, annotateTime = false;
			double sigma, beta, div;
			int half_patch;
			int half_window;
			int patch_size = 3;
			/*
			The existence of unlabeled voxels in the output image  ($\hat{L}$) requires a regularization step to assign a meaningful value to them. These cases are rare in comparison with the volume dimension ($|NaN| \ll  N^2$).
			We have assigned to the unlabeled voxels the value of the median calculated from its neighbourhood, but, for example, inpainting approaches could perform well too. The size of this neighbourhood is $\mathcal{N'}_{\mathbf{x}}$.
			The regularization operation reads as follows:\\
			\hat{L}(\mathbf{x})=\left\{
			\begin{array}{ll}
			\hat{L}(\mathbf{x}), \quad \textrm{if} \quad \hat{L}(\mathbf{x}) \neq NaN
			\\ [0.3cm]
			median_{\mathcal{N'}_{\mathbf{x}}}( \hat{L}(\mathbf{x})), \quad \textrm{otherwise}
			\end{array}
			\right.
			*/
			bool cleanNaNFlag = true;
			bool normalizeFlag = true;

			float elapsedTime;

			cudaEvent_t     start, stop;
			cudaError_t error;

			ofstream myfile, myfile2;
			cudaDeviceProp deviceProp;
			int cudaDevice ;
			sigma = 1;
			beta = 1;
			cudaDevice = 0;
			if (argc >= 8) {
				cudaDevice = atoi(argv[7]);
			}
			if (argc >= 7) {
				sigma = atof(argv[5]);
				beta = atof(argv[6]);
			}
			anatomyAtlas = argv[1];
			resultName = argv[2];
			EXEC_CHOICE choice = (EXEC_CHOICE)atoi(argv[3]);
			nh = atoi(argv[4]);



		
			cudaSetDevice(cudaDevice);
			cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, cudaDevice);

			if (error_id == cudaSuccess)
			{
				printf(" Device %d: %s\n", cudaDevice, deviceProp.name);
			}
			float *labeled_img = NULL;
			int many = 10;
			std::cout << "Arguments " << argc << std::endl;
			std::cout << "[2] Anatomy Atlas " << anatomyAtlas << std::endl;
			std::cout << "[3] Result " << resultName << std::endl;
			std::cout << "[3] Choice " << GetChoice(choice) << std::endl;
			std::cout << "[4] NH " << nh << std::endl;		
			std::cout << "[5] sigma " << sigma << std::endl;
			std::cout << "[6] beta " << beta << std::endl;
			std::cout << "[7] CudaDevice " << cudaDevice << std::endl;
			std::cout << "normalizeFlag=" << normalizeFlag << " cleanNaNFlag=" << cleanNaNFlag << std::endl;
			elements = nx *ny *nz;
			float minLabels, maxLabels;

			if (normalizeFlag) {
				elementsAnatomyAtlas = nx*ny*nz*num;
				
				for (int i = 0; i < elementsAnatomyAtlas; i++) {
					label[i] = 255.0f*((label[i] + 1024.0f) / 4095.0f);
				}

				minLabels = *std::min_element(label, label + elementsAnatomyAtlas);
				maxLabels = *std::max_element(label, label + elementsAnatomyAtlas);

				std::cout << "maxLabels " << maxLabels << "minLabels " << minLabels << std::endl;
			}


			cudaEventCreate(&start);
			cudaEventCreate(&stop);
		
			switch (choice) {
		
			case EXEC_CHOICE::GpuTestTime:
				div = (2 * 27 * beta*pow(sigma, 2));
				labeled_img = (float *)malloc(sizeof(float) *nx*ny*nz);
				memset(labeled_img, 0, sizeof(float) *nx*ny*nz);
				// PatchBasedPseudoCT D:\Users\ealcain\DB_Images\JRTIP\02_CT_Propagation\Subject_01_AtlasSize_18.mat 3 0 Subject_01_AtlasSize_18_Op2_Rd_7Nh 7 7 2.5 1 0
				for (int choiceIter =  EXEC_CHOICE::SM; choiceIter == 2; choiceIter--) {
					for (int nhIter = 7; nhIter <= 7; nhIter += 2) {						
						nh = nhIter;
						printf("####Choice %d\n", choice_op);
						//for (int atlas = 1; atlas <= 18; atlas++) {
						for (int at = 2; at >= 1; at--) {
							num = at;
							for (int i = 0; i < many; i++) {
								/******* STARTS CLOCK *************************************/
								cudaEventRecord(start);

								for (int j = 0; j < many; j++) {
									MultiPatchBasedSegmentationGpu(im, atlas, label, labeled_img, (EXEC_CHOICE) choiceIter, div);
								}
								cudaEventRecord(stop, 0);
								cudaEventSynchronize(stop);
								// get stop time, and display the timing results
								error = cudaEventElapsedTime(&elapsedTime, start, stop);
								if (error == cudaSuccess) {
									elapsedTime = elapsedTime / (float)many;

									//myfile.open("D:\\Users\\ealcain\\resultsSM_9_0.txt", ios::app);
									//myfile << " Iter  " << i << " Atlas " << atlas << " Choice " << choice_op << " Nh  " << nh << " Elapsed Time  " << elapsedTime << " ms " << (elapsedTime / 60000.0f) << " minutes\n";
									//myfile.close();
									printf("Iter %d Atlas %d Choice %d Nh %d Elapsed Time %f ms %f minutes\n", i, num, choice_op, nh, elapsedTime, elapsedTime / 60000.0f);
									//timeAtlas[atlas] += (60 * elapsedTime / 60000.0f);

									it = resultsMap.find(nhIter);
									if (it != resultsMap.end()) {
										it2 = it->second.find(at);
										if (it2 != it->second.end()) {
											it2->second.push_back(elapsedTime);
										}
										else {
											std::vector<float> vect;
											vect.push_back(elapsedTime);
											it->second.insert(std::pair<int, std::vector<float>>(at, vect));
										}
									}
									else {
										std::map<int, std::vector<float>> mapNh;
										std::vector<float> vect;
										vect.push_back(elapsedTime);
										mapNh.insert(std::pair<int, std::vector<float>>(at, vect));
										resultsMap.insert(std::pair<int, std::map<int, std::vector<float>>>(nhIter, mapNh));
									}

								}
								else {
									fprintf(stderr, "Failed to measure the time (error code %s)!\n", cudaGetErrorString(error));
									exit(EXIT_FAILURE);
								}
							}
							
							

							myfile.open(partial_total_results_path.c_str(), ios::app);
							myfile2.open(partial_results_path.c_str(), ios::app);
							float sum = 0.0f;
							myfile2 << "Nh\t" << nhIter << " Atlas\t" << at << "\n";
							for (int i = 0; i < resultsMap[nhIter][at].size(); i++) {
								sum += resultsMap[nhIter][at][i];
								myfile2 << i << " Elapsed Time\t" << resultsMap[nhIter][at][i] << "\t" << (resultsMap[nhIter][at][i] / 1000.0f) << "\t" << (resultsMap[nhIter][at][i] / 60000.0f) << "\n";
							}
							sum = sum / (float)it2->second.size();
							myfile << "Nh\t" << nhIter << " Atlas\t" << atlas << " Elapsed Time\t" << sum << "\t" << (sum / 1000.0f) << "\t" << (sum / 60000.0f) << "\n";




							myfile.close();
							myfile2.close();

						}
						myfile.open(final_results_path.c_str(), ios::app);
						myfile << "Num of executions to calculate a result " << many << " sampled " << many << " times\n";
						for (it = resultsMap.begin(); it != resultsMap.end(); ++it) {
							for (it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
								float sum = 0.0f;
								for (int i = 0; i < it2->second.size(); i++) {
									sum += it2->second[i];
								}
								sum = sum / (float)it2->second.size();
								myfile << "Nh\t" << it->first << " Atlas\t" << it2->first << " Elapsed Time\t" << sum << "\t" << (sum / 1000.0f) << "\t" << (sum / 60000.0f) << "\n";
							}
						}

						//printf("Iter %d Atlas %d Choice %d Nh %d Elapsed Time %f ms %f minutes\n", i, num, choice_op, nh, elapsedTime, elapsedTime / 60000.0f);
						//timeAtlas[atlas] += (60 * elapsedTime / 60000.0f);
						//for (int i = 0; i < 3; i++) {
						//	myfile << " Atlas " << atlas << " Choice " << choice_op << " Nh  " << nh << " Elapsed Time  " << elapsedTime << " ms " << (elapsedTime / 60000.0f) << " minutes\n";
						//}
						myfile.close();
					}
				}

				myfile.open("D:\\Users\\ealcain\\finalresultsSM_9_0.txt", ios::app);
				myfile << "Num of executions to calculate a result " << many << " sampled " << many << " times\n";
				for (it = resultsMap.begin(); it != resultsMap.end(); ++it) {
					for (it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
						float sum = 0.0f;
						for (int i = 0; i < it2->second.size(); i++) {
							sum += it2->second[i];
						}
						sum = sum / (float)it2->second.size();
						myfile << "Nh\t" << it->first << " Atlas\t" << it2->first << " Elapsed Time\t" << sum << "\t" << (sum / 1000.0f) << "\t" << (sum / 60000.0f) << "\n";
					}
				}

				//printf("Iter %d Atlas %d Choice %d Nh %d Elapsed Time %f ms %f minutes\n", i, num, choice_op, nh, elapsedTime, elapsedTime / 60000.0f);
				//timeAtlas[atlas] += (60 * elapsedTime / 60000.0f);
				//for (int i = 0; i < 3; i++) {
				//	myfile << " Atlas " << atlas << " Choice " << choice_op << " Nh  " << nh << " Elapsed Time  " << elapsedTime << " ms " << (elapsedTime / 60000.0f) << " minutes\n";
				//}
				myfile.close();
				break;
				case EXEC_CHOICE::GM:
				case EXEC_CHOICE::GM2:
				case EXEC_CHOICE::SM:
				cudaEventRecord(start);
				labeled_img = (float *)malloc(sizeof(float) *nx*ny*nz);
				div = (2 * 27 * beta*pow(sigma, 2));
				memset(labeled_img, 0, sizeof(float) *nx*ny*nz);
				
				MultiPatchBasedSegmentationGpu(im, atlas, label, labeled_img,  choice, div);

				if (labeled_img != NULL) {
					char * matFile;

					matFile = (char*)malloc(strlen(resultName.c_str()) + 5);

					sprintf(matFile, "%s.mat", resultName.c_str());

					if (cleanNaNFlag) {
						std::cout << "Regularization step " << std::endl;

						half_patch = (patch_size - 1) / 2;
						half_window = (nh - 1) / 2;

						RegularizationStep(half_window, half_patch, nx, ny, nz, labeled_img);
						RoundMatrix(labeled_img, elements);
						minLabels = 1024.0;
						maxLabels = 4095.0;

						for (int i = 0; i < elements; i++) {
							labeled_img[i] = round((maxLabels * (labeled_img[i] / 255.0f))) - minLabels;
						}
					}
					
					if (writeResult)
#ifdef MATLAB 
						WriteOutputMatFile(matFile, resultName.c_str(), labeled_img, nx, ny, nz);
#endif
					free(labeled_img);
				}
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				// get stop time, and display the timing results
				error = cudaEventElapsedTime(&elapsedTime, start, stop);
				if (error == cudaSuccess) {
					rB = sizeof(float)* (nx - ((nh - 1) / 2))*(ny - ((nh - 1) / 2))*(nz - ((nh - 1) / 2))* (num*nh * nh * nh + (2 * num * 27 * nh * nh * nh));
					wB = (nx - ((nh - 1) / 2))*(ny - ((nh - 1) / 2))*(nz - ((nh - 1) / 2)) * sizeof(float);
					div = (60.0f* (elapsedTime / 60000.0f))*1.0e9;
					
					bw_eff = (rB + wB) / div;
					printf("Effective BandWidth = %.4f GB/s, Elapsed Time %f ms %f seconds\n", bw_eff, elapsedTime, 60.0f* (elapsedTime / 60000.0f));
				}
				else {
					fprintf(stderr, "Failed to measure the time (error code %s)!\n", cudaGetErrorString(error));
					exit(EXIT_FAILURE);
				}
				break;
			}
			free(im);
			free(atlas);
			free(label);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		}

	}
	else {
		result = 0;
		std::cout << "Usage: PatchBasedPseudoCT Data.mat EXEC_CHOICE NH [sigma] [beta] " << std::endl;
		
		std::cout << "[1] Data.mat Anatomy atlas I, \mathcal{A} =\{(\mathcal{I}^{i},L^{i})\}^n_{i=1}. Atlas dimension is inferred from the data" << std::endl;
		std::cout << "[2] Name for the mat file result " << std::endl;
		std::cout << "[3] EXEC_CHOICE {GM = 0 Global Memory, GM2 = 1, Patches of input image in registers , SM = 2, shared memoryGpuTestTime = 4} " << std::endl;
		std::cout << "[4] NH {7,9,11} " << std::endl;	
		std::cout << "[5] sigma 1 default (2.5) optional only with GM/GM2/SM. sigma is the standard deviation of the noise in the images given by Signal Noise Ratio (SNR), but we can expect that images have a good SNR." << std::endl;
		std::cout << "[6] beta 1 default optional only with GM/GM2/SM. Beta is a positive real number that influences the difficulty to accept patches with less or more similarity" << std::endl;
		std::cout << "[7] cuda device default 0" << std::endl;

	}
	
#ifdef WINDOWS
	GetLocalTime(lpSystemTime);
	printf("Local Time Ends %d:%d:%d\n", lpSystemTime->wHour, lpSystemTime->wMinute, lpSystemTime->wSecond);
	free(lpSystemTime);
#else
#endif

	// In order to be able to run the profiler
	cudaDeviceReset();
	return (result == 0) ? EXIT_SUCCESS : EXIT_FAILURE;

}

