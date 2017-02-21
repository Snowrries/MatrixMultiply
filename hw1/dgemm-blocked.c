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
	double a1, a2, a3, a4, a5, a6, a7, a8;
	int i, j, k;
	//Expand j, k, then i. (Maybe have to alter, if this is column major? )
	for (i = 0; i < M; ++i) {

		for (k = 0; k < (K - 7); k += 8) {
			
			a1 = A[ lda*k+i];
			a2 = A[ lda*(k + 1)+i];
			a3 = A[ lda*(k + 2)+i];
			a4 = A[ lda*(k + 3)+i];
			a5 = A[ lda*(k + 4)+i];
			a6 = A[ lda*(k + 5)+i];
			a7 = A[ lda*(k + 6)+i];
			a8 = A[ lda*(k + 7)+i];

			for (j = 0; j < (N - 7); j += 8) {

				C[lda*j + i] += a1 * B[ lda*j + k];
				C[lda*j + i] += a2 * B[ lda*j + k + 1];
				C[lda*j + i] += a3 * B[ lda*j + k + 2];
				C[lda*j + i] += a4 * B[ lda*j + k + 3];
				C[lda*j + i] += a5 * B[ lda*j + k + 4];
				C[lda*j + i] += a6 * B[ lda*j + k + 5];
				C[lda*j + i] += a7 * B[ lda*j + k + 6];
				C[lda*j + i] += a8 * B[ lda*j + k + 7];

				C[lda*(j+1) + i] += a1 * B[ lda*( j+1 ) + k];
				C[lda*(j+1) + i] += a2 * B[ lda*( j+1 ) + k + 1];
				C[lda*(j+1) + i] += a3 * B[ lda*( j+1 ) + k + 2];
				C[lda*(j+1) + i] += a4 * B[ lda*( j+1 ) + k + 3];
				C[lda*(j+1) + i] += a5 * B[ lda*( j+1 ) + k + 4];
				C[lda*(j+1) + i] += a6 * B[ lda*( j+1 ) + k + 5];
				C[lda*(j+1) + i] += a7 * B[ lda*( j+1 ) + k + 6];
				C[lda*(j+1) + i] += a8 * B[ lda*( j+1 ) + k + 7];


				C[lda*(j+2) + i] += a1 * B[ lda*( j+2 ) + k];
				C[lda*(j+2) + i] += a2 * B[ lda*( j+2 ) + k + 1];
				C[lda*(j+2) + i] += a3 * B[ lda*( j+2 ) + k + 2];
				C[lda*(j+2) + i] += a4 * B[ lda*( j+2 ) + k + 3];
				C[lda*(j+2) + i] += a5 * B[ lda*( j+2 ) + k + 4];
				C[lda*(j+2) + i] += a6 * B[ lda*( j+2 ) + k + 5];
				C[lda*(j+2) + i] += a7 * B[ lda*( j+2 ) + k + 6];
				C[lda*(j+2) + i] += a8 * B[ lda*( j+2 ) + k + 7];


				C[lda*(j+3) + i] += a1 * B[ lda*( j+3 ) + k];
				C[lda*(j+3) + i] += a2 * B[ lda*( j+3 ) + k + 1];
				C[lda*(j+3) + i] += a3 * B[ lda*( j+3 ) + k + 2];
				C[lda*(j+3) + i] += a4 * B[ lda*( j+3 ) + k + 3];
				C[lda*(j+3) + i] += a5 * B[ lda*( j+3 ) + k + 4];
				C[lda*(j+3) + i] += a6 * B[ lda*( j+3 ) + k + 5];
				C[lda*(j+3) + i] += a7 * B[ lda*( j+3 ) + k + 6];
				C[lda*(j+3) + i] += a8 * B[ lda*( j+3 ) + k + 7];


				C[lda*(j+4) + i] += a1 * B[ lda*( j+4 ) + k];
				C[lda*(j+4) + i] += a2 * B[ lda*( j+4 ) + k + 1];
				C[lda*(j+4) + i] += a3 * B[ lda*( j+4 ) + k + 2];
				C[lda*(j+4) + i] += a4 * B[ lda*( j+4 ) + k + 3];
				C[lda*(j+4) + i] += a5 * B[ lda*( j+4 ) + k + 4];
				C[lda*(j+4) + i] += a6 * B[ lda*( j+4 ) + k + 5];
				C[lda*(j+4) + i] += a7 * B[ lda*( j+4 ) + k + 6];
				C[lda*(j+4) + i] += a8 * B[ lda*( j+4 ) + k + 7];


				C[lda*(j+5) + i] += a1 * B[ lda*( j+5 ) + k];
				C[lda*(j+5) + i] += a2 * B[ lda*( j+5 ) + k + 1];
				C[lda*(j+5) + i] += a3 * B[ lda*( j+5 ) + k + 2];
				C[lda*(j+5) + i] += a4 * B[ lda*( j+5 ) + k + 3];
				C[lda*(j+5) + i] += a5 * B[ lda*( j+5 ) + k + 4];
				C[lda*(j+5) + i] += a6 * B[ lda*( j+5 ) + k + 5];
				C[lda*(j+5) + i] += a7 * B[ lda*( j+5 ) + k + 6];
				C[lda*(j+5) + i] += a8 * B[ lda*( j+5 ) + k + 7];


				C[lda*(j+6) + i] += a1 * B[ lda*( j+6 ) + k];
				C[lda*(j+6) + i] += a2 * B[ lda*( j+6 ) + k + 1];
				C[lda*(j+6) + i] += a3 * B[ lda*( j+6 ) + k + 2];
				C[lda*(j+6) + i] += a4 * B[ lda*( j+6 ) + k + 3];
				C[lda*(j+6) + i] += a5 * B[ lda*( j+6 ) + k + 4];
				C[lda*(j+6) + i] += a6 * B[ lda*( j+6 ) + k + 5];
				C[lda*(j+6) + i] += a7 * B[ lda*( j+6 ) + k + 6];
				C[lda*(j+6) + i] += a8 * B[ lda*( j+6 ) + k + 7];


				C[lda*(j+7) + i] += a1 * B[ lda*( j+7 ) + k];
				C[lda*(j+7) + i] += a2 * B[ lda*( j+7 ) + k + 1];
				C[lda*(j+7) + i] += a3 * B[ lda*( j+7 ) + k + 2];
				C[lda*(j+7) + i] += a4 * B[ lda*( j+7 ) + k + 3];
				C[lda*(j+7) + i] += a5 * B[ lda*( j+7 ) + k + 4];
				C[lda*(j+7) + i] += a6 * B[ lda*( j+7 ) + k + 5];
				C[lda*(j+7) + i] += a7 * B[ lda*( j+7 ) + k + 6];
				C[lda*(j+7) + i] += a8 * B[ lda*( j+7 ) + k + 7];


			}	
			if(N % 8){
				for (; j < N; ++j) {
					C[lda*j + i] += a1 * B[ lda*j + k];
					C[lda*j + i] += a2 * B[ lda*j + k + 1];
					C[lda*j + i] += a3 * B[ lda*j + k + 2];
					C[lda*j + i] += a4 * B[ lda*j + k + 3];
					C[lda*j + i] += a5 * B[ lda*j + k + 4];
					C[lda*j + i] += a6 * B[ lda*j + k + 5];
					C[lda*j + i] += a7 * B[ lda*j + k + 6];
					C[lda*j + i] += a8 * B[ lda*j + k + 7];
				}
			}
		}
		if (K % 8) {
			do {
				a1 = A[ lda*k+i];
				for (j = 0; j < N; ++j) {

					C[lda*j + i] += a1 * B[ lda*j + k];
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
