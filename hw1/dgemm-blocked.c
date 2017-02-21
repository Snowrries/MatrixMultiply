/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include "immintrin.h"
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C)
{
	double b1, b2, b3, b4, b5, b6, b7, b8;
	double c1, c2, c3, c4, c5, c6, c7, c8;
	int i, j, k;
	//Expand j, k, then i. (Maybe have to alter, if this is column major? )
	for (i = 0; i < M; ++i) {

		for (k = 0; k < (K - 7); k += 8) {

			restrict b1 = B[ lda*i + k];
			restrict b2 = B[ lda*i + k + 1];
			restrict b3 = B[ lda*i + k + 2];
			restrict b4 = B[ lda*i + k + 3];
			restrict b5 = B[ lda*i + k + 4];
			restrict b6 = B[ lda*i + k + 5];
			restrict b7 = B[ lda*i + k + 6];
			restrict b8 = B[ lda*i + k + 7];

			for (j = 0; j < (N - 7); j += 8) {

				restrict c1 = C[lda*i + j];
				restrict c2 = C[lda*i + (j+1)];
				restrict c3 = C[lda*i + (j+2)];
				restrict c4 = C[lda*i + (j+3)];
				restrict c5 = C[lda*i + (j+4)];
				restrict c6 = C[lda*i + (j+5)];
				restrict c7 = C[lda*i + (j+6)];
				restrict c8 = C[lda*i + (j+7)];


				c1 += restrict A[ lda*k+j] * b1;
				c2 += restrict A[ lda*k+(j+1)] * b1;
				c3 += restrict A[ lda*k+(j+2)] * b1;
				c4 += restrict A[ lda*k+(j+3)] * b1;
				c5 += restrict A[ lda*k+(j+4)] * b1;
				c6 += restrict A[ lda*k+(j+5)] * b1;
				c7 += restrict A[ lda*k+(j+6)] * b1;
				c8 += restrict A[ lda*k+(j+7)] * b1;

				c1 += restrict A[ lda*(k + 1)+j] * b2;
				c2 += restrict A[ lda*(k + 1)+(j+1)] * b2;
				c3 += restrict A[ lda*(k + 1)+(j+2)] * b2;
				c4 += restrict A[ lda*(k + 1)+(j+3)] * b2;
				c5 += restrict A[ lda*(k + 1)+(j+4)] * b2;
				c6 += restrict A[ lda*(k + 1)+(j+5)] * b2;
				c7 += restrict A[ lda*(k + 1)+(j+6)] * b2;
				c8 += restrict A[ lda*(k + 1)+(j+7)] * b2;

				c1 += restrict A[ lda*(k + 2)+j] * b3;
				c2 += restrict A[ lda*(k + 2)+(j+1)] * b3;
				c3 += restrict A[ lda*(k + 2)+(j+2)] * b3;
				c4 += restrict A[ lda*(k + 2)+(j+3)] * b3;
				c5 += restrict A[ lda*(k + 2)+(j+4)] * b3;
				c6 += restrict A[ lda*(k + 2)+(j+5)] * b3;
				c7 += restrict A[ lda*(k + 2)+(j+6)] * b3;
				c8 += restrict A[ lda*(k + 2)+(j+7)] * b3;

				c1 += restrict A[ lda*(k + 3)+j] * b4;
				c2 += restrict A[ lda*(k + 3)+(j+1)] * b4;
				c3 += restrict A[ lda*(k + 3)+(j+2)] * b4;
				c4 += restrict A[ lda*(k + 3)+(j+3)] * b4;
				c5 += restrict A[ lda*(k + 3)+(j+4)] * b4;
				c6 += restrict A[ lda*(k + 3)+(j+5)] * b4;
				c7 += restrict A[ lda*(k + 3)+(j+6)] * b4;
				c8 += restrict A[ lda*(k + 3)+(j+7)] * b4;

				c1 += restrict A[ lda*(k + 4)+j] * b5;
				c2 += restrict A[ lda*(k + 4)+(j+1)] * b5;
				c3 += restrict A[ lda*(k + 4)+(j+2)] * b5;
				c4 += restrict A[ lda*(k + 4)+(j+3)] * b5;
				c5 += restrict A[ lda*(k + 4)+(j+4)] * b5;
				c6 += restrict A[ lda*(k + 4)+(j+5)] * b5;
				c7 += restrict A[ lda*(k + 4)+(j+6)] * b5;
				c8 += restrict A[ lda*(k + 4)+(j+7)] * b5;

				c1 += restrict A[ lda*(k + 5)+j] * b6;
				c2 += restrict A[ lda*(k + 5)+(j+1)] * b6;
				c3 += restrict A[ lda*(k + 5)+(j+2)] * b6;
				c4 += restrict A[ lda*(k + 5)+(j+3)] * b6;
				c5 += restrict A[ lda*(k + 5)+(j+4)] * b6;
				c6 += restrict A[ lda*(k + 5)+(j+5)] * b6;
				c7 += restrict A[ lda*(k + 5)+(j+6)] * b6;
				c8 += restrict A[ lda*(k + 5)+(j+7)] * b6;

				c1 += restrict A[ lda*(k + 6)+j] * b7;
				c2 += restrict A[ lda*(k + 6)+(j+1)] * b7;
				c3 += restrict A[ lda*(k + 6)+(j+2)] * b7;
				c4 += restrict A[ lda*(k + 6)+(j+3)] * b7;
				c5 += restrict A[ lda*(k + 6)+(j+4)] * b7;
				c6 += restrict A[ lda*(k + 6)+(j+5)] * b7;
				c7 += restrict A[ lda*(k + 6)+(j+6)] * b7;
				c8 += restrict A[ lda*(k + 6)+(j+7)] * b7;

				c1 += restrict A[ lda*(k + 7)+j] * b8;
				c2 += restrict A[ lda*(k + 7)+(j+1)] * b8;
				c3 += restrict A[ lda*(k + 7)+(j+2)] * b8;
				c4 += restrict A[ lda*(k + 7)+(j+3)] * b8;
				c5 += restrict A[ lda*(k + 7)+(j+4)] * b8;
				c6 += restrict A[ lda*(k + 7)+(j+5)] * b8;
				c7 += restrict A[ lda*(k + 7)+(j+6)] * b8;
				c8 += restrict A[ lda*(k + 7)+(j+7)] * b8;

			}	
			if(N % 8){
				for (; j < N; j ++) {
					C[lda*i + j] += A[ lda*k+j] * b1;
					C[lda*i + j] += A[ lda*(k + 1)+j] * b2;
					C[lda*i + j] += A[ lda*(k + 2)+j] * b3;
					C[lda*i + j] += A[ lda*(k + 3)+j] * b4;
					C[lda*i + j] += A[ lda*(k + 4)+j] * b5;
					C[lda*i + j] += A[ lda*(k + 5)+j] * b6;
					C[lda*i + j] += A[ lda*(k + 6)+j] * b7;
					C[lda*i + j] += A[ lda*(k + 7)+j] * b8;
				}
			}
		}
		if (K % 8) {
			do {
				b1 = B[j*lda + k];
				for (j = 0; j < N; ++j) {

					C[lda*i + j] += A[k*lda + j] * b1;
				}
			} while (++k < K);
		}
	}
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm(int lda, double* A, double* B, double* C)
{
	/* For each block-row of A */
	for (int i = 0; i < lda; i += BLOCK_SIZE)
		/* For each block-column of B */
		for (int j = 0; j < lda; j += BLOCK_SIZE)
			/* Accumulate block dgemms into block of C */
			for (int k = 0; k < lda; k += BLOCK_SIZE)
			{
				/* Correct block dimensions if block "goes off edge of" the matrix */
				int M = min(BLOCK_SIZE, lda - i);
				int N = min(BLOCK_SIZE, lda - j);
				int K = min(BLOCK_SIZE, lda - k);

				/* Perform individual block dgemm */
				do_block(lda, M, N, K, A + j + k*lda, B + k + j*lda, C + j + j*lda);
			}
}
