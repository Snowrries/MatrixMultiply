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
	double bb1, bb2, bb3, bb4;
	__m256d b1, b2, b3, b4, c;
	int i, j, k;
	//Expand j, k, then i. (Maybe have to alter, if this is column major? )
	for (j = 0; j < N; ++j) {

		for (k = 0; k < (K - 3); k += 4) {

			b1 = _mm256_broadcast_sd (&B[ lda*j + k]);
			b2 = _mm256_broadcast_sd (&B[ lda*j + (k + 1)]);
			b3 = _mm256_broadcast_sd (&B[ lda*j + (k + 2)]);
			b4 = _mm256_broadcast_sd (&B[ lda*j + (k + 3)]);
			for (i = 0; i < (M - 3); i += 4) {
				c = _mm256_loadu_pd(&C[lda*j + i]);

				c = _mm256_add_pd(c, _mm256_mul_pd(_mm256_loadu_pd(&A[lda*k + i]), b1));

				c = _mm256_add_pd(c, _mm256_mul_pd(_mm256_loadu_pd(&A[lda*(k + 1) + i]), b2));

				c = _mm256_add_pd(c, _mm256_mul_pd(_mm256_loadu_pd(&A[lda*(k + 2) + i]), b3));

				c = _mm256_add_pd(c, _mm256_mul_pd(_mm256_loadu_pd(&A[lda*(k + 3) + i]), b4));

				_mm256_storeu_pd(&C[lda*j + i],c);

			}	
			if(M % 4){
				bb1 = B[j*lda + k];
				bb2 = B[j*lda + k+1];
				bb3 = B[j*lda + k+2];
				bb4 = B[j*lda + k+3];
				for (; i < M; ++i) {				
					C[lda*j + i] += A[lda*k + i] * b1;
					C[lda*j + i] += A[lda*(k + 1) + i] * b1;
					C[lda*j + i] += A[lda*(k + 2) + i] * b1;
					C[lda*j + i] += A[lda*(k + 3) + i] * b1;
				}
			}
		}
		if (K % 4) {
			do {
				bb1 = B[j*lda + k];
				for (i = 0; i < (M - 3); i += 4) {
					C[lda*j + i] += A[lda*k + i] * b1;
					C[lda*j + (i+1)] += A[lda*k + (i+1)] * b1;
					C[lda*j + (i+2)] += A[lda*k + (i+2)] * b1;
					C[lda*j + (i+3)] += A[lda*k + (i+3)] * b1;
				}
				if(M % 4){
					for (; i < M; i ++) {				
						C[lda*j + i] += A[lda*k + i] * b1;
					}
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
