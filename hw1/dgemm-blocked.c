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
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C)
{
	double b1, b2, b3, b4, b5, b6, b7, b8;
	//Expand j, k, then i. (Maybe have to alter, if this is column major? )
	for (int j = 0; j < N; ++j) {

		for (int k = 0; k < (K - 7); k += 8) {

			b1 = B[ lda*j + k];
			b2 = B[ lda*j + k + 1];
			b3 = B[ lda*j + k + 2];
			b4 = B[ lda*j + k + 3];
			b5 = B[ lda*j + k + 4];
			b6 = B[ lda*j + k + 5];
			b7 = B[ lda*j + k + 6];
			b8 = B[ lda*j + k + 7];

			for (int i = 0; i < M; ++i) {
				C[ lda*j + i] += A[lda*k + i] * b1;
				C[ lda*j + i] += A[lda*(k + 1) + i] * b2;
				C[ lda*j + i] += A[lda*(k + 2) + i] * b3;
				C[ lda*j + i] += A[lda*(k + 3) + i] * b4;
				C[ lda*j + i] += A[lda*(k + 4) + i] * b5;
				C[ lda*j + i] += A[lda*(k + 5) + i] * b6;
				C[ lda*j + i] += A[lda*(k + 6) + i] * b7;
				C[ lda*j + i] += A[lda*(k + 7) + i] * b8;

				if (K % 8) {
					do {
						b1 = B[j*lda + k];
						for (i = 0; i < M; ++i) {
							C[j*lda + i] += A[k*lda + i] * b1;
						}
					} while (++k < K);
				}
			}

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
