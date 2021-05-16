/*
* pseudo-CT estimation based on multi-atlas with patches on mutlicore and manycore platforms. For further details see [1,2]
* If you use this software for research purposes, YOU MUST CITE the corresponding
* of the following papers in any resulting publication:
* 
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017). https://doi.org/10.1007/s11554-016-0568-0 [2] Angel Torrado-Carvajal, Joaquin L. Herraiz, Eduardo Alcain, Antonio S. Montemayor, Lina Garcia-Cañamaque, Juan A. Hernandez-Tamames, Yves Rozenholc and Norberto Malpica Journal of Nuclear Medicine January 2016, 57 (1) 136-143; DOI: https://doi.org/10.2967/jnumed.115.156299
* [2] Angel Torrado-Carvajal, Joaquin L Herraiz, Eduardo Alcain, Antonio S Mon-temayor, Lina Garcia-Cañamaque, Juan A Hernandez-Tamames, Yves Rozen-holc, and Norberto Malpica. “Fast Patch-Based Pseudo-CT Synthesis fromT1-Weighted MR Images for PET/MR Attenuation Correction in Brain Stud-ies”. In:Journal of nuclear medicine : official publication, Society of Nuclear Medicine57.1 (Jan. 2016), pp. 136–143.ISSN: 0161-5505.DOI:10.2967/jnumed.115.156299.
* * PatchBasedPseudoCT is free software: you can redistribute it and/or modify
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




#include  <math.h>
#include <stdio.h>
#ifdef WINDOWS
#include <Windows.h>
#endif // WINDOWS
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <chrono>
#include <vector>
#include <omp.h>
#include <float.h>
#include <algorithm>

#include <time.h>       /* time */
//#define DEBUG_ALG

// https://docs.microsoft.com/en-us/cpp/build/reference/d-preprocessor-definitions?view=vs-2019
#ifdef MATLAB 
#pragma message( "MATLAB Compiled")
#include "mat.h"
#else 
#pragma message( "Random data")
#endif
#include <iostream>
#include <fstream>
using namespace std;
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

/* 4D Matrix from Matlab
2x3x4x2

*/
// Image
float * im;
float *atlas;
// Label
float *label;
// Size of the image
int nx, ny, nz;
int num, nh;

typedef enum
{
	SingleCore = 0,
	MultiCore = 1,
	SingleCoreTestTime = 2,
	MultiCoreTestTime = 3,
	
} EXEC_CHOICE;

// ********************** StopWatch ******************************************//
/* Simple wrapper class to measure the time through chrono */
//http://en.cppreference.com/w/cpp/chrono/duration
class StopWatch
{
public:
	enum TimeUnit
	{
		NANOSECONDS, MICROSECONDS, MILLISECONDS, SECONDS, MINUTES, HOURS
	};
	StopWatch(void);
	~StopWatch(void);

	void Start();
	void Stop();
	double GetElapsedTime(TimeUnit timeUnit);


private:
	const double SECOND_TO_MILLIS = 1000;
	const double MINUTE_TO_SECONDS = 60;
	const double HOUR_TO_MINUTES = 60;
	// // wraps QueryPerformanceCounter
	std::chrono::high_resolution_clock::time_point _startingTime;
	std::chrono::high_resolution_clock::time_point _endingTime;

};

StopWatch::StopWatch() {

}

StopWatch::~StopWatch() {

}
/** Start measuring the time
*/
void StopWatch::Start() {
	_startingTime = std::chrono::high_resolution_clock::now();
}

/** Stop measuring the time
*/
void StopWatch::Stop() {
	_endingTime = std::chrono::high_resolution_clock::now();
}
//http://en.cppreference.com/w/cpp/chrono/duration/duration_cast

/** Stop GetElapsedTime the time
*  @param timeUnit: unit to calculate the time elapsed
*  @param return: time in ms, nano seconds, microseconds etc.
*/
double StopWatch::GetElapsedTime(TimeUnit timeUnit) {
	double elapsedTime;
	//http://en.cppreference.com/w/cpp/numeric/ratio/ratio
	std::chrono::duration<double, std::milli> fp_ms;
	std::chrono::duration<double, std::nano> fp_nano;
	std::chrono::duration<double, std::micro> fp_mc;
	std::chrono::duration<double, std::deca> fp_deca;
	switch (timeUnit)
	{
	case NANOSECONDS:
		fp_nano = _endingTime - _startingTime;
		elapsedTime = fp_nano.count();
		break;
	case MICROSECONDS:
		fp_mc = _endingTime - _startingTime;
		elapsedTime = fp_mc.count();
		break;
	case MILLISECONDS:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = fp_ms.count();
		break;
	case SECONDS:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = (double)fp_ms.count() / SECOND_TO_MILLIS;
		break;
	case MINUTES:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = (double)fp_ms.count() / (SECOND_TO_MILLIS * MINUTE_TO_SECONDS);
		break;
	case HOURS:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = (double)fp_ms.count() / (SECOND_TO_MILLIS * MINUTE_TO_SECONDS * HOUR_TO_MINUTES);
		break;
	default:
		elapsedTime = -1;
		break;
	}

	return elapsedTime;
}

// ********************** StopWatch ******************************************//

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
	void Store(std::string fileName, const T* vect,const int num,const int rows,const int cols,const int depth);

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
void MatrixIO<T>::Store(std::string fileName, const T* vect,const int num,const int rows,const int cols,const int depth) {
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


// Functions for Reading Matlab Data
#ifdef MATLAB 
int WriteOutputMatFile(const char *file, const char *variable, const float *data, const int nx, const int ny, const int nz);
void ConvertMatrixMatlab(const float *matrix, float * matrix_matlab, const int nx, const int ny, const int nz);
void AnalyzeStructArray(const mxArray *sPtr, const char *fName, float *matrix, int item, int items);
void ConvertToRowBasedII(const mxArray *pa, float *matrix, int item, int items);
float* ConvertToRowBased(const mxArray *pa);
int Choice(const char * name);
bool FindDimension(const char *file);
bool CheckDimension(const mxArray *sPtr, const char *fName, int & dimX, int & dimY, int & dimZ);
#endif
// Debug Functions 
void PrintMatrix(char * msg, matrixMask maskIm, float * matrix);
void PrintMatrix(char * msg, float * matrix, int size);
void PrintMatrix(char * msg, float * matrix, int size, int value);
void printMatrixMask(char * msg, matrixMask mask);
// Synthesise the new modality
double PatchSimilarity(matrixMask maskIm, matrixMask maskAtlas, const float * im, const float *atlas, int offset, double div);

void MultiAtlasPatchBasedSegmentation(const float * img, const float *atlas, const float *atlas_label, float *labeled_img, const double div);
void MultiAtlasPatchBasedSegmentationOpenMP(const float * img, const float *atlas, const float *atlas_label, float *labeled_img, const double div);
// Regularization step
double Median(const float *patchImg, const int length);
void RegularizationStep(const int half_window, const int half_patch, const int nx, const int ny, const int nz, float *segmentationArray);
void RoundMatrix(float * mat,const int size);
// Read input data 
int ReadMatFile(const char *file);


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
float* ConvertToRowBased( const mxArray *pa) {
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
void ConvertToRowBasedII(const mxArray *pa, float *matrix,const int item, const int items) {
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
	printf("size_array %d r=%d c=%d z=%d %d\n", size_array, r,c,z, size_array==r*c*z );
	printf("size_array %d nx=%d ny=%d nz=%d %d\n", size_array, nx,ny,nz, size_array==nx*ny*nz);

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

 bool CheckDimension(const mxArray *sPtr, const char *fName, int & dimX , int & dimY , int & dimZ) {
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
		std::cout << "Error closing file " <<  file <<std::endl;
		return(1);
	}
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		std::cout << "Error reopening file "<< file << std::endl;
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
			std::cout << "Struct " <<  mxGetNumberOfFields(pa)<< std::endl;
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
			
				valid = valid && CheckDimension(pa, mxGetFieldNameByNumber(pa, ii), nx,ny,nz);
			}
		}
		else {
			

			const  mwSize *size = mxGetDimensions(pa);

			if (mxGetNumberOfDimensions(pa) == 3) {
				if (nx != size[0] || ny != size[1] || nz != size[2]){
					std::cout << "Name " << name << " do NOT match the dimension "<< std::endl;
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
void ConvertMatrixMatlab(const float *matrix, float * matrix_matlab,const int nx, const int ny, const int nz) {
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
int WriteOutputMatFile(const char *file, const char *variable,const float *data,const int nx,const int ny,const int nz) {
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
		std::cout << __FILE__ <<": Out of memory on line " << __LINE__ << std::endl;
		std::cout << "Unable to create mxArray " << file << std::endl;
		return(EXIT_FAILURE);
	}

	memcpy((void *)(mxGetPr(pa1)), (void *)data_matlab, size * sizeof(float));
	status = matPutVariable(pmat, variable, pa1);
	if (status != 0) {
		std::cout << __FILE__ << " :  Error using matPutVariable on line " <<  __LINE__ << std::endl;
		return(EXIT_FAILURE);
	}

	/* clean up */
	mxDestroyArray(pa1);


	if (matClose(pmat) != 0) {
		std::cout << "Error closing file" <<file << std::endl;
		return(EXIT_FAILURE);
	}


	std::cout << "File " << file << " written" << std::endl;
	return(EXIT_SUCCESS);
}

#else
// http://www.cplusplus.com/doc/tutorial/files/
int readMatFile(const char *file) {
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
			int elements = nx * ny * nz * num;
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
			MatrixIO<float> matrixIO;
			matrixIO.store("atlas.txt", atlas, num, nx, ny, nz);
			matrixIO.store("label.txt", label, num, nx, ny, nz);
			matrixIO.store("im.txt", im, 1, nx, ny, nz);
		}
		else {
			std::cout << "Reading data" <<std::endl;
			MatrixIO<float> matrixIO;
			vector<int> dimensionsAtlas;
			vector<int> dimensionsLabel;
			vector<int> dimensionsIm;

			matrixIO.read_(atlasStr, atlas, dimensionsAtlas);
			matrixIO.read_(labelStr, label, dimensionsLabel);
			matrixIO.read_(imStr, im, dimensionsIm);

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
				std::cout << "Dimensions do not match" <<std::endl;
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
						
					}

				}
			}
		}
	}
	float numNanFlt = numNaN;
	float total = nx *ny *nz;
	std::cout << "NumLops " << numLops << " NumNaN " << numNaN << " % " << 100.0f *(numNanFlt / total) << " NumMax " << numMax << std::endl;
}




/** Print the mask values from the matrix
*  @param msg: name of the mask variable
*  @param maskIm: limits of the patch
*  @param matrix: data
*
*/
void PrintMatrix(const char * msg,const matrixMask maskIm,const float * matrix) {
	std::cout << msg <<std::endl;

	for (int k = maskIm.z; k <= maskIm.zZ; k++) {
		for (int i = maskIm.r; i <= maskIm.rR; i++) {
			for (int j = maskIm.c; j <= maskIm.cC; j++) {
				printf("%.10e\n", matrix[(k*maskIm.rows*maskIm.cols) + (i*maskIm.cols) + j]);
			}
		}

	}

}
/** Print matrix mask
*  @param msg: name of the mask variable
*  @param matrix: data
*  @param size: limits of the patch
*
*/
void PrintMatrix(const char * msg,const float * matrix,const int size) {
	std::cout << msg << std::endl;
	for (int i = 0; i< size; i++) {
		printf("%.10e\n", matrix[i]);
	}

}
/** Print matrix mask
*  @param msg: name of the mask variable
*  @param matrix: data
*  @param size: limits of the patch
*  @param value: distinct of value and 0.0f
*
*/
void PrintMatrix(const char * msg,const float * matrix,const int size,const int value) {
	std::cout << msg << " Elements " << size << " Value " << value << std::endl;
	for (int i = 0; i< size; i++) {
		if (matrix[i] != value && matrix[i] != 0.0f) {
			printf("i %d %.10e\n", i, matrix[i]);
		}
	}

}
/** Print matrix mask
*  @param msg: name of the mask variable
*  @param mask: limits of the patch
*/
void PrintMatrixMask(const char * msg,const matrixMask mask) {

	printf("%s z %d zZ %d c %d  cC %d  r %d rR %d rows %d cols %d mats\n", msg, mask.z, mask.zZ, mask.c, mask.cC, mask.r, mask.rR, mask.rows, mask.cols, mask.mats);
}
/** Measure the similarity between two patches see [3]
* div = (2 * N*beta*pow(sigma, 2))
* ms = (patch - patch_at). ^ 2;
* arg = sum(ms(:)) / (2 * N*beta*pow(sigma, 2));
* weight = exp(-arg);
*  @param maskIm: limits of the image patch
*  @param maskAtlas: limits of the atlas patch
*  @param im: input image MRI
*  @param atlas: MRI image from the anatomy atlas
*  @param offset: shift in the anatomy atlas (all the atlas are flatten in one dimension array)
*  @param div: see formula (1) 2S beta sigma^2 in [1]
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017).
* [3] Pierrick Coupé, José V. Manjón, Vladimir Fonov, Jens Pruessner, Montserrat
Robles, and D. Louis Collins. “Patch-based segmentation using expert priors:
Application to hippocampus and ventricle segmentation”. In: NeuroImage
54.2 (Jan. 2011), pp. 940–954. DOI: 10.1016/j.neuroimage.2010.09.018.
*/
double PatchSimilarity(const matrixMask maskIm, const matrixMask maskAtlas, const float * im, const float *atlas,const int offset,const double div) {
	double patch = 0;
	double patch_at = 0;
	double sumRet = 0;
	double arg, weight;
#ifdef  DEBUG_ALG
	std::cout << "IM PATCH" << maskIm << std::endl;
	std::cout << "ATLAS PATCH" << maskAtlas << std::endl;
	
#endif //  DEBUG_ALG

	
	int pitchIm = maskIm.rows*maskIm.cols;
	int pitchAtlas = maskAtlas.rows*maskAtlas.cols;
	for (int k = maskIm.z, kk = maskAtlas.z; (k <= maskIm.zZ) && (kk <= maskAtlas.zZ); k++, kk++) {
		for (int i = maskIm.r, ii = maskAtlas.r; (i <= maskIm.rR) && (ii <= maskAtlas.rR); i++, ii++) {
			for (int j = maskIm.c, jj = maskAtlas.c; (j <= maskIm.cC) && (jj <= maskAtlas.cC); j++, jj++) {
				patch = im[(k*pitchIm) + (i*maskIm.cols) + j];
				patch_at = atlas[offset + (kk*pitchAtlas) + (ii*maskAtlas.cols) + jj];
				sumRet += pow(patch - patch_at, 2);				
			}
		}
	}
	//printf("sumRet %.10e N %f\n", sumRet,N);
	arg = sumRet / div;
	//printf("ARG %.10e\n", arg);
	weight = exp(-arg);
	return weight;
}




/** Round an array 
*  @param mat: array to round 
*  @param size: number of elements of the array
*/
void RoundMatrix(float * mat,const int size) {
	for (int i = 0; i < size; i++) {
		//printf("round(mat[i]) %f ", round(mat[i]));
		mat[i] = round(mat[i]);
		//printf(" mat[i] %f \n", mat[i]);
	}
}
/** Synthesise a new modality from an input image using an anatomy atlas. An anatomy atlas is a pair of aligned volumes {MRI,CT}
*  Using OpenMP multicore, set omp_set_num_threads(num_threads) through command parameters
*  @param img: input image for example MRI volume
*  @param atlas: atlas images for example MRI
*  @param atlas_label: labels for the modality to build in the atlas
*  @param labeled_img: synthetic image for the new modality
*  @param div: see formula (1) 2S beta sigma^2 in [1]
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017).
https://doi.org/10.1007/s11554-016-0568-0
*/
void MultiAtlasPatchBasedSegmentation(const float * img, const float *atlas, const float *atlas_label, float *labeled_img, const double div) {

	int patch_size = 3;
	int window = nh;
	int half_patch = (patch_size - 1) / 2;
	int half_window = (window - 1) / 2;

	int i = 0;
	int left_x, rigth_x, left_y, rigth_y, left_z, rigth_z, left_u, rigth_u, left_v, rigth_v, left_w, rigth_w;


	double label, weight;
	matrixMask maskIm, maskAtlas;
	int size_matrix = window*window*window;
	int pitch = nx*ny*nz;
	int offset = 0;

	double labeled_img_value = 0;
	double product = 0, sum_weight = 0;
	int countNaN = 0;

	//int x,y,z,u,v,w;

	std::cout << "Nh " << window << std::endl;

	for (int z = half_window + half_patch; z< nz - half_window - half_patch; z++) {
		for (int x = half_window + half_patch; x< nx - half_window - half_patch; x++) {
			for (int y = half_window + half_patch; y < ny - half_window - half_patch; y++) {


				i++;


				// Matlab code conversion
				//patch = img(left_x:rigth_x,left_y:rigth_y,left_z:rigth_z);
				maskIm.c = y - half_patch;
				maskIm.cC = y + half_patch;
				maskIm.r = x - half_patch;
				maskIm.rR = x + half_patch;
				maskIm.z = z - half_patch;
				maskIm.zZ = z + half_patch;
				maskIm.cols = ny;
				maskIm.rows = nx;
				maskIm.mats = nz;



				product = 0; sum_weight = 0;

				for (int at = 0; at<num; at++) {
					offset = pitch * at;
					for (int w = z - half_window; w <= z + half_window; w++) {
						for (int u = x - half_window; u <= x + half_window; u++) {

							for (int v = y - half_window; v <= y + half_window; v++) {




								//printf("neighbourhood [%d,%d] [%d,%d] [%d,%d]\n" , left_u,rigth_u,left_v,rigth_v,left_w,rigth_w);   
								//patch_at = atlas(left_u:rigth_u,left_v:rigth_v,left_w:rigth_w);
								//printf("atlas_label [%d,%d,%d] %d\n",u,v,w,(w*nx*ny) + (u*ny) +v );
								label = atlas_label[offset + (w*nx*ny) + (u*ny) + v];

								maskAtlas.c = v - half_patch;
								maskAtlas.cC = v + half_patch;
								maskAtlas.r = u - half_patch;
								maskAtlas.rR = u + half_patch;
								maskAtlas.z = w - half_patch;
								maskAtlas.zZ = w + half_patch;
								maskAtlas.cols = ny;
								maskAtlas.rows = nx;
								maskAtlas.mats = nz;

								//ms = (patch-patch_at).^2;
								//arg = sum(ms(:))/(2*N*beta*pow(sigma,2));
								//weight = exp(-arg);

								weight = PatchSimilarity(maskIm, maskAtlas, im, atlas, offset, div);

								product += weight * label;
								sum_weight += weight;

							}
						}
					}
				}
				if (abs(product) > DBL_EPSILON && abs(sum_weight) >DBL_EPSILON)

					labeled_img[(z*nx*ny) + (x*ny) + y] = product / sum_weight;
				else {
					labeled_img[(z*nx*ny) + (x*ny) + y] = NAN;
					countNaN++;
				}

			}
		}
	}
	std::cout << "NaN " << countNaN << std::endl;
}


/** Synthesise a new modality from an input image using an anatomy atlas. An anatomy atlas is a pair of aligned volumes {MRI,CT}
*  Using OpenMP multicore, set omp_set_num_threads(num_threads) through command parameters
*  @param img: input image for example MRI volume
*  @param atlas: atlas images for example MRI
*  @param atlas_label: labels for the modality to build in the atlas 
*  @param labeled_img: synthetic image for the new modality
*  @param div: see formula (1) 2S beta sigma^2 in [1]
* [1] Alcaín, E., Torrado-Carvajal, A., Montemayor, A.S. et al. Real-time patch-based medical image modality propagation by GPU computing. J Real-Time Image Proc 13, 193–204 (2017).
https://doi.org/10.1007/s11554-016-0568-0 
*/
void MultiAtlasPatchBasedSegmentationOpenMP(const float * img, const float *atlas, const float *atlas_label, float *labeled_img, const double div) {

	int patch_size = 3;
	int window = nh;
	int half_patch = (patch_size - 1) / 2;
	int half_window = (window - 1) / 2;

	int i = 0;
	int left_x, rigth_x, left_y, rigth_y, left_z, rigth_z, left_u, rigth_u, left_v, rigth_v, left_w, rigth_w;


	double label, weight;
	matrixMask maskIm, maskAtlas;
	int size_matrix = window*window*window;
	int pitch = nx*ny*nz;
	int offset = 0;

	double labeled_img_value = 0;
	double product = 0, sum_weight = 0;
	int countNaN = 0;

	//int x,y,z,u,v,w;

	std::cout << "Nh " << window << std::endl;
#pragma omp parallel for private(product, sum_weight, label, weight,maskIm,  maskAtlas,offset) 
	for (int z = half_window + half_patch; z< nz - half_window - half_patch; z++) {
		for (int x = half_window + half_patch; x< nx - half_window - half_patch; x++) {
			for (int y = half_window + half_patch; y < ny - half_window - half_patch; y++) {


				i++;
			

				// Matlab code conversion
				//patch = img(left_x:rigth_x,left_y:rigth_y,left_z:rigth_z);
				maskIm.c = y - half_patch;
				maskIm.cC = y + half_patch;
				maskIm.r = x - half_patch;
				maskIm.rR = x + half_patch;
				maskIm.z = z - half_patch;
				maskIm.zZ = z + half_patch;
				maskIm.cols = ny;
				maskIm.rows = nx;
				maskIm.mats = nz;



				product = 0; sum_weight = 0;

				for (int at = 0; at<num; at++) {
					offset = pitch * at;
					for (int w = z - half_window; w <= z + half_window; w++) {
						for (int u = x - half_window; u <= x + half_window; u++) {

							for (int v = y - half_window; v <= y + half_window; v++) {



								
								//printf("neighbourhood [%d,%d] [%d,%d] [%d,%d]\n" , left_u,rigth_u,left_v,rigth_v,left_w,rigth_w);   
								//patch_at = atlas(left_u:rigth_u,left_v:rigth_v,left_w:rigth_w);
								//printf("atlas_label [%d,%d,%d] %d\n",u,v,w,(w*nx*ny) + (u*ny) +v );
								label = atlas_label[offset + (w*nx*ny) + (u*ny) + v];

								maskAtlas.c = v - half_patch;
								maskAtlas.cC = v + half_patch;
								maskAtlas.r = u - half_patch;
								maskAtlas.rR = u + half_patch;
								maskAtlas.z = w - half_patch;
								maskAtlas.zZ = w + half_patch;
								maskAtlas.cols = ny;
								maskAtlas.rows = nx;
								maskAtlas.mats = nz;

								//ms = (patch-patch_at).^2;
								//arg = sum(ms(:))/(2*N*beta*pow(sigma,2));
								//weight = exp(-arg);

								weight = PatchSimilarity(maskIm, maskAtlas, im, atlas, offset, div);

								product += weight * label;
								sum_weight += weight;

							}
						}
					}
				}
				if (abs(product) > DBL_EPSILON && abs(sum_weight) >DBL_EPSILON)

					labeled_img[(z*nx*ny) + (x*ny) + y] = product / sum_weight;
				else {
					labeled_img[(z*nx*ny) + (x*ny) + y] = NAN;
					countNaN++;
				}

			}
		}
	}
	std::cout << "NaN " << countNaN << std::endl;
}




/** Get the string of a choice
*
*  @param choice: choice to display
*  @return depending on the choice a string with SingleCore, MultiCore, SingleCoreTestTime, MultiCoreTestTime
*/
string GetChoice(const EXEC_CHOICE choice) {
	string choiceStr;
	switch (choice) {


	case EXEC_CHOICE::SingleCore:
		choiceStr = "SingleCore";
		break;
	case EXEC_CHOICE::MultiCore:
		choiceStr = "MultiCore";
		break;
	case EXEC_CHOICE::SingleCoreTestTime:
		choiceStr = "SingleCoreTestTime";
		break;
	case EXEC_CHOICE::MultiCoreTestTime:
		choiceStr = "MultiCoreTestTime";
		break;
	default:
		choiceStr = "Unknown";
		break;
	}

		return choiceStr;
}

// NewPatchedBrainPureC Data.mat EXEC_CHOICE NH [TH] [sigma] [beta]
//	- [1] Data.mat Anatomy atlas I, \mathcal{A} =\{(\mathcal{I}^{i},L^{i})\}^n_{i=1}. Atlas dimension is inferred from the data
//  - [2] Name for the mat file result
//	- [3] EXEC_CHOICE {SingleCore = 0, MultiCore = 1, SingleCoreTestTime = 2, MultiCoreTestTime = 3, } 
//  - [4] NH {7,9,11}
//	- [5] TH Number of threads optional only with MultiCore
//	- [6] sigma 1 default (2.5) optional only with MultiCore/SingleCore. sigma is the standard deviation of the noise in the images given by Signal Noise Ratio (SNR), but we can expect that images have a good SNR.
//	- [7] beta 1 default optional only with MultiCore/SingleCore. Beta is a positive real number that influences the difficulty to accept patches with less or more similarity


//	
// Examples
// SingleCore Nh=7
// NewPatchedBrainPureC data\Subject_08_AtlasSize_01.mat Subject_08_AtlasSize_01_7Nh_SC 0 7 
// Milticore  Nh=7 Th=12
// NewPatchedBrainPureC data\Subject_08_AtlasSize_01.mat Subject_08_AtlasSize_01_7Nh_MC_TH12 0 7 12
// Milticore  Nh=7 Th=12 sigma=2.5 beta=1.0
// NewPatchedBrainPureC data\Subject_08_AtlasSize_01.mat Subject_08_AtlasSize_01_7Nh_MC_TH12 0 7 12 2.5 1
int main(int argc, char **argv)
{


	int result;
	float elapsedTime;
#ifdef WINDOWS
	LPSYSTEMTIME lpSystemTime;

	lpSystemTime = (LPSYSTEMTIME)malloc(sizeof(SYSTEMTIME));
	GetLocalTime(lpSystemTime);
	std::cout <<"Local Time Starts " << lpSystemTime->wHour <<":"<< lpSystemTime->wMinute << ":" <<lpSystemTime->wSecond << std:endl;
#else

#endif
	if (argc >=4) {
		ofstream myfile;
		bool checkValidData = FindDimension(argv[1]);
		if (checkValidData) {
			std::cout << "nx=" << nx << " ny=" << ny << " nz=" << nz << " num=" << num << std::endl;

			result = ReadMatFile(argv[1]);
			int num_threads, elementsAnatomyAtlas, elements;

			string anatomyAtlas;
			string resultName;
			string pathFile = "D:\\Users\\ealcain\\new_results_cpu_222_222_112.txt";
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


			StopWatch stopWatch;

			num_threads = 1;
			sigma = 1;
			beta = 1;
			if (argc >= 6) {
				num_threads = atoi(argv[5]);
			}
			if (argc >= 8) {
				sigma = atof(argv[6]);
				beta = atof(argv[7]);
			}
			anatomyAtlas = argv[1];
			resultName = argv[2];
			EXEC_CHOICE choice = (EXEC_CHOICE)atoi(argv[3]);
			nh = atoi(argv[4]);


			//http://stackoverflow.com/questions/14337278/precise-time-measurement add #include <Windows.h>
			// http://msdn.microsoft.com/en-us/library/windows/desktop/ms644904%28v=vs.85%29.aspx
			float *labeled_img = NULL, *labeled_img_prev = NULL;
			int many = 10;
			std::cout << "[2] Anatomy Atlas " << anatomyAtlas << std::endl;
			std::cout << "[3] Result " << resultName << std::endl;
			std::cout << "[3] Choice " << GetChoice(choice) << std::endl;
			std::cout << "[4] NH " << nh << std::endl;
			std::cout << "[5] TH " << num_threads << std::endl;
			std::cout << "[6] sigma " << sigma << std::endl;
			std::cout << "[7] beta " << beta << std::endl;
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


			switch (choice) {


			case EXEC_CHOICE::SingleCore:

				div = (2 * 27 * beta*pow(sigma, 2));

				std::cout << "MultiPatchBasedSegmentation div=" << div << std::endl;
				labeled_img = (float *)malloc(sizeof(float) *nx*ny*nz);

				memset(labeled_img, 0, sizeof(float) *nx*ny*nz);

				// start timer
				stopWatch.Start();

				MultiAtlasPatchBasedSegmentation(im, atlas, label, labeled_img, div);

				// stop timer
				stopWatch.Stop();

				elapsedTime = stopWatch.GetElapsedTime(StopWatch::MILLISECONDS);
				std::cout << "Atlas " << num << " Nh " << nh << " Elapsed Time " << elapsedTime << " ms " << " minutes " << elapsedTime / 60000.0f << std::endl;
				if (annotateTime) {

					myfile.open(pathFile, ios::app);
					myfile << "Atlas  " << num << " Nh " << nh << " Elapsed Time " << elapsedTime << " ms  " << elapsedTime / 60000.0f << " minutes  \n";
					myfile.close();
				}
				break;

			case EXEC_CHOICE::MultiCore:

				omp_set_num_threads(num_threads);

				div = (2 * 27 * beta*pow(sigma, 2));

				std::cout << "MultiPatchBasedSegmentationOmp div=" << div << std::endl;
				labeled_img = (float *)malloc(sizeof(float) *nx*ny*nz);

				memset(labeled_img, 0, sizeof(float) *nx*ny*nz);
				// start timer
				stopWatch.Start();

				MultiAtlasPatchBasedSegmentationOpenMP(im, atlas, label, labeled_img, div);

				// stop timer
				stopWatch.Stop();

				elapsedTime = stopWatch.GetElapsedTime(StopWatch::MILLISECONDS);


				std::cout << "Atlas " << num << " Nh " << nh << " Elapsed Time " << elapsedTime << " ms " << " minutes " << elapsedTime / 60000.0f << std::endl;
				if (annotateTime) {

					myfile.open(pathFile, ios::app);
					myfile << "Atlas  " << num << " Nh " << nh << " Elapsed Time " << elapsedTime << " ms  " << elapsedTime / 60000.0f << " minutes  \n";
					myfile.close();
				}

				break;




			case EXEC_CHOICE::SingleCoreTestTime:


				std::cout << "SingleCoreTestTime div=" << div << std::endl;
				labeled_img = (float *)malloc(sizeof(float) *nx*ny*nz);
				div = (2 * 27 * beta*pow(sigma, 2));
				memset(labeled_img, 0, sizeof(float) *nx*ny*nz);
				for (int i = 0; i < many; i++) {

					// start timer
					stopWatch.Start();

					for (int j = 0; j < many; j++) {
						MultiAtlasPatchBasedSegmentation(im, atlas, label, labeled_img, div);
					}
					// stop timer

					stopWatch.Stop();

					elapsedTime = stopWatch.GetElapsedTime(StopWatch::MILLISECONDS);

					elapsedTime = elapsedTime / (float)many;
					std::cout << "Iter " << i << " Elapsed Time " << elapsedTime << " ms " << " minutes " << elapsedTime / 60000.0f << std::endl;
				}
				break;
			case EXEC_CHOICE::MultiCoreTestTime:
				num_threads = atoi(argv[3]);
				std::cout << "MultiCoreTestTime num_threads " << num_threads;
				omp_set_num_threads(num_threads);
				div = (2 * 27 * beta*pow(sigma, 2));

				labeled_img = (float *)malloc(sizeof(float) *nx*ny*nz);

				memset(labeled_img, 0, sizeof(float) *nx*ny*nz);

				for (int i = 0; i < many; i++) {

					// start timer
					stopWatch.Start();
					for (int j = 0; j < many; j++) {
						MultiAtlasPatchBasedSegmentationOpenMP(im, atlas, label, labeled_img, div);
					}
					// stop timer
					stopWatch.Stop();

					elapsedTime = stopWatch.GetElapsedTime(StopWatch::MILLISECONDS);

					elapsedTime = elapsedTime / (float)many;
					std::cout << "Iter " << i << " Elapsed Time " << elapsedTime << " ms " << " minutes " << elapsedTime / 60000.0f << std::endl;

				}
				break;





			}
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
				std::cout << "matfile " << resultName << " Matfile " << matFile << std::endl;
				if (writeResult)
#ifdef MATLAB 
					WriteOutputMatFile(matFile, resultName.c_str(), labeled_img, nx, ny, nz);
#else
					printf("Saving MatFile %s\n", argv[4]);
				MatrixIO<float> matrixIO;
				matrixIO.store(matFile, labeled_img, 1, nx, ny, nz);
#endif
				free(labeled_img);
			}
			free(im);
			free(atlas);
			free(label);
		}

		
	}
	else {
		result = 0;
		std::cout << "Usage: PatchBasedSynthesis Data.mat EXEC_CHOICE NH [TH] [sigma] [beta] " << std::endl;
		// NewPatchedBrainPureC 
		std::cout << "[1] Data.mat Anatomy atlas I, \mathcal{A} =\{(\mathcal{I}^{i},L^{i})\}^n_{i=1}. Atlas dimension is inferred from the data" << std::endl;
		std::cout << "[2] Name for the mat file result "<< std::endl;
		std::cout << "[3] EXEC_CHOICE {SingleCore = 0, MultiCore = 1, SingleCoreTestTime = 2, MultiCoreTestTime = 3, } "<< std::endl;
		std::cout << "[4] NH {7,9,11} "<< std::endl;
		std::cout << "[5] TH Number of threads optional only with MultiCore " << std::endl;
		std::cout << "[6] sigma 1 default (2.5) optional only with MultiCore/SingleCore. sigma is the standard deviation of the noise in the images given by Signal Noise Ratio (SNR), but we can expect that images have a good SNR." << std::endl;
		std::cout << "[7] beta 1 default optional only with MultiCore/SingleCore. Beta is a positive real number that influences the difficulty to accept patches with less or more similarity" << std::endl;

		
	}
#ifdef WINDOWS
	GetLocalTime(lpSystemTime);
	printf("Local Time Ends %d:%d:%d\n", lpSystemTime->wHour, lpSystemTime->wMinute, lpSystemTime->wSecond);
	free(lpSystemTime);
#else
#endif
	return (result == 0) ? EXIT_SUCCESS : EXIT_FAILURE;


}


