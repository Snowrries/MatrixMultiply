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
	int k1,k2,k3,k4,k5,k6,k7,k8;
	int i, j, k;
	//Expand j, k, then i. (Maybe have to alter, if this is column major? )
	for (j = 0; j < N; ++j) {

		for (k = 0; k < (K - 7); k += 8) {

			b1 = B[ lda*j + k];
			b2 = B[ lda*j + k + 1];
			b3 = B[ lda*j + k + 2];
			b4 = B[ lda*j + k + 3];
			b5 = B[ lda*j + k + 4];
			b6 = B[ lda*j + k + 5];
			b7 = B[ lda*j + k + 6];
			b8 = B[ lda*j + k + 7];

			for (i = 0; i < (M-7); i += 8) {

				k1 = lda*k;
				k2 = lda*(k + 1);
				k3 = lda*(k + 2);
				k4 = lda*(k + 3);
				k5 = lda*(k + 4);
				k6 = lda*(k + 5);
				k7 = lda*(k + 6);
				k8 = lda*(k + 7);

				C[lda*j + i] += A[k1 + i] * b1;
				C[lda*j + i] += A[k2 + i] * b2;
				C[lda*j + i] += A[k3 + i] * b3;
				C[lda*j + i] += A[k4 + i] * b4;
				C[lda*j + i] += A[k5 + i] * b5;
				C[lda*j + i] += A[k6 + i] * b6;
				C[lda*j + i] += A[k7 + i] * b7;
				C[lda*j + i] += A[k8 + i] * b8;

				C[lda*j + (i + 1)] += A[k1 + (i + 1)] * b1;
				C[lda*j + (i + 1)] += A[k2 + (i + 1)] * b2;
				C[lda*j + (i + 1)] += A[k3 + (i + 1)] * b3;
				C[lda*j + (i + 1)] += A[k4 + (i + 1)] * b4;
				C[lda*j + (i + 1)] += A[k5 + (i + 1)] * b5;
				C[lda*j + (i + 1)] += A[k6 + (i + 1)] * b6;
				C[lda*j + (i + 1)] += A[k7 + (i + 1)] * b7;
				C[lda*j + (i + 1)] += A[k8 + (i + 1)] * b8;

				C[lda*j + (i + 2)] += A[k1 + (i + 2)] * b1;
				C[lda*j + (i + 2)] += A[k2 + (i + 2)] * b2;
				C[lda*j + (i + 2)] += A[k3 + (i + 2)] * b3;
				C[lda*j + (i + 2)] += A[k4 + (i + 2)] * b4;
				C[lda*j + (i + 2)] += A[k5 + (i + 2)] * b5;
				C[lda*j + (i + 2)] += A[k6 + (i + 2)] * b6;
				C[lda*j + (i + 2)] += A[k7 + (i + 2)] * b7;
				C[lda*j + (i + 2)] += A[k8 + (i + 2)] * b8;

				C[lda*j + (i + 3)] += A[k1 + (i + 3)] * b1;
				C[lda*j + (i + 3)] += A[k2 + (i + 3)] * b2;
				C[lda*j + (i + 3)] += A[k3 + (i + 3)] * b3;
				C[lda*j + (i + 3)] += A[k4 + (i + 3)] * b4;
				C[lda*j + (i + 3)] += A[k5 + (i + 3)] * b5;
				C[lda*j + (i + 3)] += A[k6 + (i + 3)] * b6;
				C[lda*j + (i + 3)] += A[k7 + (i + 3)] * b7;
				C[lda*j + (i + 3)] += A[k8 + (i + 3)] * b8;

				C[lda*j + (i + 4)] += A[k1 + (i + 4)] * b1;
				C[lda*j + (i + 4)] += A[k2 + (i + 4)] * b2;
				C[lda*j + (i + 4)] += A[k3 + (i + 4)] * b3;
				C[lda*j + (i + 4)] += A[k4 + (i + 4)] * b4;
				C[lda*j + (i + 4)] += A[k5 + (i + 4)] * b5;
				C[lda*j + (i + 4)] += A[k6 + (i + 4)] * b6;
				C[lda*j + (i + 4)] += A[k7 + (i + 4)] * b7;
				C[lda*j + (i + 4)] += A[k8 + (i + 4)] * b8;

				C[lda*j + (i + 5)] += A[k1 + (i + 5)] * b1;
				C[lda*j + (i + 5)] += A[k2 + (i + 5)] * b2;
				C[lda*j + (i + 5)] += A[k3 + (i + 5)] * b3;
				C[lda*j + (i + 5)] += A[k4 + (i + 5)] * b4;
				C[lda*j + (i + 5)] += A[k5 + (i + 5)] * b5;
				C[lda*j + (i + 5)] += A[k6 + (i + 5)] * b6;
				C[lda*j + (i + 5)] += A[k7 + (i + 5)] * b7;
				C[lda*j + (i + 5)] += A[k8 + (i + 5)] * b8;

				C[lda*j + (i + 6)] += A[k1 + (i + 6)] * b1;
				C[lda*j + (i + 6)] += A[k2 + (i + 6)] * b2;
				C[lda*j + (i + 6)] += A[k3 + (i + 6)] * b3;
				C[lda*j + (i + 6)] += A[k4 + (i + 6)] * b4;
				C[lda*j + (i + 6)] += A[k5 + (i + 6)] * b5;
				C[lda*j + (i + 6)] += A[k6 + (i + 6)] * b6;
				C[lda*j + (i + 6)] += A[k7 + (i + 6)] * b7;
				C[lda*j + (i + 6)] += A[k8 + (i + 6)] * b8;

				C[lda*j + (i + 7)] += A[k1 + (i + 7)] * b1;
				C[lda*j + (i + 7)] += A[k2 + (i + 7)] * b2;
				C[lda*j + (i + 7)] += A[k3 + (i + 7)] * b3;
				C[lda*j + (i + 7)] += A[k4 + (i + 7)] * b4;
				C[lda*j + (i + 7)] += A[k5 + (i + 7)] * b5;
				C[lda*j + (i + 7)] += A[k6 + (i + 7)] * b6;
				C[lda*j + (i + 7)] += A[k7 + (i + 7)] * b7;
				C[lda*j + (i + 7)] += A[k8 + (i + 7)] * b8;
				}
			if (M % 8) {
				do {
					C[lda*j + i] += A[k1 + i] * b1;
					C[lda*j + i] += A[k2 + i] * b2;
					C[lda*j + i] += A[k3 + i] * b3;
					C[lda*j + i] += A[k4 + i] * b4;
					C[lda*j + i] += A[k5 + i] * b5;
					C[lda*j + i] += A[k6 + i] * b6;
					C[lda*j + i] += A[k7 + i] * b7;
					C[lda*j + i] += A[k8 + i] * b8;
				} while (++i < M);
			}
		}
		if (K % 8) {
			do {
				b1 = B[j*lda + k];
				for (i = 0; i < (M-7); i += 8) {

					C[lda*j + i] += A[k1 + i] * b1;
					C[lda*j + (i + 1)] += A[lda*k + (i + 1)] * b1;
					C[lda*j + (i + 2)] += A[lda*k + (i + 2)] * b1;
					C[lda*j + (i + 3)] += A[lda*k + (i + 3)] * b1;
					C[lda*j + (i + 4)] += A[lda*k + (i + 4)] * b1;
					C[lda*j + (i + 5)] += A[lda*k + (i + 5)] * b1;
					C[lda*j + (i + 6)] += A[lda*k + (i + 6)] * b1;
					C[lda*j + (i + 7)] += A[lda*k + (i + 7)] * b1;
				}
				if (M % 8) {
					do {
						C[lda*j + i] += A[lda*k + i] * b1;
					} while (++i < M);
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
				do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
			}
}
