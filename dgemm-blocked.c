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
	int i, j, k;
	//Expand j, k, then i. (Maybe have to alter, if this is column major? )
	for (i = 0; i < M; ++i) {

		for (k = 0; k < (K - 7); k += 8) {

			b1 = B[ lda*i + k];
			b2 = B[ lda*i + k + 1];
			b3 = B[ lda*i + k + 2];
			b4 = B[ lda*i + k + 3];
			b5 = B[ lda*i + k + 4];
			b6 = B[ lda*i + k + 5];
			b7 = B[ lda*i + k + 6];
			b8 = B[ lda*i + k + 7];

			for (j = 0; j < (N - 7); j += 8) {

				C[lda*i + j] += A[ lda*k+j] * b1;
				C[lda*i + j] += A[ lda*(k + 1)+j] * b2;
				C[lda*i + j] += A[ lda*(k + 2)+j] * b3;
				C[lda*i + j] += A[ lda*(k + 3)+j] * b4;
				C[lda*i + j] += A[ lda*(k + 4)+j] * b5;
				C[lda*i + j] += A[ lda*(k + 5)+j] * b6;
				C[lda*i + j] += A[ lda*(k + 6)+j] * b7;
				C[lda*i + j] += A[ lda*(k + 7)+j] * b8;

				C[lda*i + (j+1)] += A[ lda*k+(j+1)] * b1;
				C[lda*i + (j+1)] += A[ lda*(k + 1)+(j+1)] * b2;
				C[lda*i + (j+1)] += A[ lda*(k + 2)+(j+1)] * b3;
				C[lda*i + (j+1)] += A[ lda*(k + 3)+(j+1)] * b4;
				C[lda*i + (j+1)] += A[ lda*(k + 4)+(j+1)] * b5;
				C[lda*i + (j+1)] += A[ lda*(k + 5)+(j+1)] * b6;
				C[lda*i + (j+1)] += A[ lda*(k + 6)+(j+1)] * b7;
				C[lda*i + (j+1)] += A[ lda*(k + 7)+(j+1)] * b8;
				

				C[lda*i + (j+2)] += A[ lda*k+(j+2)] * b1;
				C[lda*i + (j+2)] += A[ lda*(k + 1)+(j+2)] * b2;
				C[lda*i + (j+2)] += A[ lda*(k + 2)+(j+2)] * b3;
				C[lda*i + (j+2)] += A[ lda*(k + 3)+(j+2)] * b4;
				C[lda*i + (j+2)] += A[ lda*(k + 4)+(j+2)] * b5;
				C[lda*i + (j+2)] += A[ lda*(k + 5)+(j+2)] * b6;
				C[lda*i + (j+2)] += A[ lda*(k + 6)+(j+2)] * b7;
				C[lda*i + (j+2)] += A[ lda*(k + 7)+(j+2)] * b8;
				

				C[lda*i + (j+3)] += A[ lda*k+(j+3)] * b1;
				C[lda*i + (j+3)] += A[ lda*(k + 1)+(j+3)] * b2;
				C[lda*i + (j+3)] += A[ lda*(k + 2)+(j+3)] * b3;
				C[lda*i + (j+3)] += A[ lda*(k + 3)+(j+3)] * b4;
				C[lda*i + (j+3)] += A[ lda*(k + 4)+(j+3)] * b5;
				C[lda*i + (j+3)] += A[ lda*(k + 5)+(j+3)] * b6;
				C[lda*i + (j+3)] += A[ lda*(k + 6)+(j+3)] * b7;
				C[lda*i + (j+3)] += A[ lda*(k + 7)+(j+3)] * b8;
				

				C[lda*i + (j+4)] += A[ lda*k+(j+4)] * b1;
				C[lda*i + (j+4)] += A[ lda*(k + 1)+(j+4)] * b2;
				C[lda*i + (j+4)] += A[ lda*(k + 2)+(j+4)] * b3;
				C[lda*i + (j+4)] += A[ lda*(k + 3)+(j+4)] * b4;
				C[lda*i + (j+4)] += A[ lda*(k + 4)+(j+4)] * b5;
				C[lda*i + (j+4)] += A[ lda*(k + 5)+(j+4)] * b6;
				C[lda*i + (j+4)] += A[ lda*(k + 6)+(j+4)] * b7;
				C[lda*i + (j+4)] += A[ lda*(k + 7)+(j+4)] * b8;
				

				C[lda*i + (j+5)] += A[ lda*k+(j+5)] * b1;
				C[lda*i + (j+5)] += A[ lda*(k + 1)+(j+5)] * b2;
				C[lda*i + (j+5)] += A[ lda*(k + 2)+(j+5)] * b3;
				C[lda*i + (j+5)] += A[ lda*(k + 3)+(j+5)] * b4;
				C[lda*i + (j+5)] += A[ lda*(k + 4)+(j+5)] * b5;
				C[lda*i + (j+5)] += A[ lda*(k + 5)+(j+5)] * b6;
				C[lda*i + (j+5)] += A[ lda*(k + 6)+(j+5)] * b7;
				C[lda*i + (j+5)] += A[ lda*(k + 7)+(j+5)] * b8;
				

				C[lda*i + (j+6)] += A[ lda*k+(j+6)] * b1;
				C[lda*i + (j+6)] += A[ lda*(k + 1)+(j+6)] * b2;
				C[lda*i + (j+6)] += A[ lda*(k + 2)+(j+6)] * b3;
				C[lda*i + (j+6)] += A[ lda*(k + 3)+(j+6)] * b4;
				C[lda*i + (j+6)] += A[ lda*(k + 4)+(j+6)] * b5;
				C[lda*i + (j+6)] += A[ lda*(k + 5)+(j+6)] * b6;
				C[lda*i + (j+6)] += A[ lda*(k + 6)+(j+6)] * b7;
				C[lda*i + (j+6)] += A[ lda*(k + 7)+(j+6)] * b8;
				

				C[lda*i + (j+7)] += A[ lda*k+(j+7)] * b1;
				C[lda*i + (j+7)] += A[ lda*(k + 1)+(j+7)] * b2;
				C[lda*i + (j+7)] += A[ lda*(k + 2)+(j+7)] * b3;
				C[lda*i + (j+7)] += A[ lda*(k + 3)+(j+7)] * b4;
				C[lda*i + (j+7)] += A[ lda*(k + 4)+(j+7)] * b5;
				C[lda*i + (j+7)] += A[ lda*(k + 5)+(j+7)] * b6;
				C[lda*i + (j+7)] += A[ lda*(k + 6)+(j+7)] * b7;
				C[lda*i + (j+7)] += A[ lda*(k + 7)+(j+7)] * b8;

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
				do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
			}
}
