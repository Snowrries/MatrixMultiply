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
	double bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10, bb11, bb12, bb13, bb14, bb15, bb16;
	int temp, t1,t2,t3,t4;
	__m256d b1, b2, b3, b4, c, temp1, temp2, temp3, temp4, temp5, temp6;
	__m256d b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16;
	int i, j, k;
	//Expand j, k, then i. (Maybe have to alter, if this is column major? )
	for (j = 0; j < N-3 ; j+=4 ) {

		for (k = 0; k < (K - 3); k += 4) {
			temp = lda*j + k; 
			b1 = _mm256_broadcast_sd (&B[ temp ]);
			b2 = _mm256_broadcast_sd (&B[ temp + 1]);
			b3 = _mm256_broadcast_sd (&B[ temp + 2]);
			b4 = _mm256_broadcast_sd (&B[ temp + 3]);
			temp += lda;
			b5 = _mm256_broadcast_sd (&B[ temp ]);
			b6 = _mm256_broadcast_sd (&B[ temp + 1]);
			b7 = _mm256_broadcast_sd (&B[ temp + 2]);
			b8 = _mm256_broadcast_sd (&B[ temp + 3]);
			temp += lda;
			b9 = _mm256_broadcast_sd (&B[ temp ]);
			b10 = _mm256_broadcast_sd (&B[ temp + 1]);
			b11 = _mm256_broadcast_sd (&B[ temp + 2]);
			b12 = _mm256_broadcast_sd (&B[ temp + 3]);
			temp += lda;
			b13 = _mm256_broadcast_sd (&B[ temp ]);
			b14 = _mm256_broadcast_sd (&B[ temp + 1]);
			b15 = _mm256_broadcast_sd (&B[ temp + 2]);
			b16 = _mm256_broadcast_sd (&B[ temp + 3]);
			for (i = 0; i < (M - 3); i += 4) {
				c = _mm256_load_pd(&C[lda*j + i]);
				t1 = lda*k+i;
				t2 = lda*(k+1)+i;
				t3 = lda*(k+2)+i;
				t4 = lda*(k+3)+i;
				temp1 = _mm256_mul_pd(_mm256_load_pd(&A[t1]), b1);
				temp2 = _mm256_mul_pd(_mm256_load_pd(&A[t2]), b2);
				temp3 = _mm256_mul_pd(_mm256_load_pd(&A[t3]), b3);
				temp4 = _mm256_mul_pd(_mm256_load_pd(&A[t4]), b4);

				temp5 = _mm256_add_pd(temp1, temp2);
				temp6 = _mm256_add_pd(temp3, temp4);

				c = _mm256_add_pd(c, temp5);
				c = _mm256_add_pd(c, temp6);

				_mm256_store_pd(&C[lda*j + i],c);

				c = _mm256_load_pd(&C[lda*(j+1) + i]);
				temp1 = _mm256_mul_pd(_mm256_load_pd(&A[t1]), b5);
				temp2 = _mm256_mul_pd(_mm256_load_pd(&A[t2]), b6);
				temp3 = _mm256_mul_pd(_mm256_load_pd(&A[t3]), b7);
				temp4 = _mm256_mul_pd(_mm256_load_pd(&A[t4]), b8);

				temp5 = _mm256_add_pd(temp1, temp2);
				temp6 = _mm256_add_pd(temp3, temp4);

				c = _mm256_add_pd(c, temp5);
				c = _mm256_add_pd(c, temp6);

				_mm256_store_pd(&C[lda*(j+1) + i],c);

				c = _mm256_load_pd(&C[lda*(j+2) + i]);
				temp1 = _mm256_mul_pd(_mm256_load_pd(&A[t1]), b9);
				temp2 = _mm256_mul_pd(_mm256_load_pd(&A[t2]), b10);
				temp3 = _mm256_mul_pd(_mm256_load_pd(&A[t3]), b11);
				temp4 = _mm256_mul_pd(_mm256_load_pd(&A[t4]), b12);

				temp5 = _mm256_add_pd(temp1, temp2);
				temp6 = _mm256_add_pd(temp3, temp4);

				c = _mm256_add_pd(c, temp5);
				c = _mm256_add_pd(c, temp6);

				_mm256_store_pd(&C[lda*(j+2) + i],c);

				c = _mm256_load_pd(&C[lda*(j+3) + i]);
				temp1 = _mm256_mul_pd(_mm256_load_pd(&A[t1]), b13);
				temp2 = _mm256_mul_pd(_mm256_load_pd(&A[t2]), b14);
				temp3 = _mm256_mul_pd(_mm256_load_pd(&A[t3]), b15);
				temp4 = _mm256_mul_pd(_mm256_load_pd(&A[t4]), b16);

				temp5 = _mm256_add_pd(temp1, temp2);
				temp6 = _mm256_add_pd(temp3, temp4);

				c = _mm256_add_pd(c, temp5);
				c = _mm256_add_pd(c, temp6);

				_mm256_store_pd(&C[lda*(j+3) + i],c);

			}	
			if(M % 4){
				temp = j*lda+k;
				bb1 = B[temp];
				bb2 = B[temp+1];
				bb3 = B[temp+2];
				bb4 = B[temp+3];
				temp += lda;
				bb5 = B[temp];
				bb6 = B[temp+1];
				bb7 = B[temp+2];
				bb8 = B[temp+3];
				temp += lda;
				bb9 = B[temp];
				bb10 = B[temp+1];
				bb11 = B[temp+2];
				bb12 = B[temp+3];
				temp += lda;
				bb13 = B[temp];
				bb14 = B[temp+1];
				bb15 = B[temp+2];
				bb16 = B[temp+3];
				for (; i < M; ++i) {
					temp = lda*k+i;
					int tempp = lda*j+i;
					C[tempp] += A[temp] * bb1;
					C[tempp] += A[temp+lda] * bb2;
					C[tempp] += A[temp+(2*lda)] * bb3;
					C[tempp] += A[temp+(3*lda)] * bb4;
					tempp+= lda;
					C[tempp] += A[temp] * bb5;
					C[tempp] += A[temp+lda] * bb6;
					C[tempp] += A[temp+(2*lda)] * bb7;
					C[tempp] += A[temp+(3*lda)] * bb8;
					tempp+= lda;
					C[tempp] += A[temp] * bb9;
					C[tempp] += A[temp+lda] * bb10;
					C[tempp] += A[temp+(2*lda)] * bb11;
					C[tempp] += A[temp+(3*lda)] * bb12;
					tempp+= lda;
					C[tempp] += A[temp] * bb13;
					C[tempp] += A[temp+lda] * bb14;
					C[tempp] += A[temp+(2*lda)] * bb15;
					C[tempp] += A[temp+(3*lda)] * bb16;
				}
			}
		}
		if (K % 4) {
			do {
				temp = lda*j + k; 
				b1 = _mm256_broadcast_sd (&B[ temp ]);
				b2 = _mm256_broadcast_sd (&B[ temp+lda ]);
				b3 = _mm256_broadcast_sd (&B[ temp+(2*lda)]);
				b4 = _mm256_broadcast_sd (&B[ temp+(3*lda)]);

				for (i = 0; i < (M - 7); i += 8) {
					c = _mm256_load_pd(&C[lda*j + i]);
					temp = lda*k+i;
					double* atp = &A[temp];
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp),b1);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*j + i],c);

					c = _mm256_load_pd(&C[lda*j + i+4]);
					int templus4 = temp+4;
					double* atp4 = &A[templus4];
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp4),b1);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*j + i+4],c);

					c = _mm256_load_pd(&C[lda*(j+1) + i]);
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp),b2);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*(j+1) + i],c);

					c = _mm256_load_pd(&C[lda*(j+1) + i+4]);
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp4),b2);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*(j+1) + i+4],c);

					c = _mm256_load_pd(&C[lda*(j+2) + i]);
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp),b3);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*(j+2) + i],c);

					c = _mm256_load_pd(&C[lda*(j+2) + i+4]);
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp4),b3);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*(j+2) + i+4],c);

					c = _mm256_load_pd(&C[lda*(j+3) + i]);
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp),b4);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*(j+3) + i],c);

					c = _mm256_load_pd(&C[lda*(j+3) + i+4]);
					temp1 = _mm256_mul_pd(_mm256_load_pd(atp4),b4);
					c = _mm256_add_pd(c, temp1);
					_mm256_store_pd(&C[lda*(j+3) + i+4],c);

				}	
				if(M % 8){
					temp = j*lda+k;
					bb1 = B[temp];
					bb2 = B[temp + lda];
					bb3 = B[temp + 2*lda];
					bb4 = B[temp + 3*lda];
					bb3 = B[temp + 4*lda];
					bb4 = B[temp + 5*lda];
					bb3 = B[temp + 6*lda];
					bb4 = B[temp + 7*lda];
					for (; i < M; ++i) {
						temp = lda*k+i;
						double at = A[temp];
						int tempp = lda*j+i;
						C[tempp] += at * bb1;
						C[tempp+ lda] += at * bb2;
						C[tempp+ 2*lda] += at * bb3;
						C[tempp+ 3*lda] += at * bb4;
					}
				}
			} while (++k < K);
		}
	}
	if( N % 4){
		do{
			for (k = 0; k < (K - 3); k += 4) {
				temp = lda*j + k; 
				b1 = _mm256_broadcast_sd (&B[ temp ]);
				b2 = _mm256_broadcast_sd (&B[ temp + 1]);
				b3 = _mm256_broadcast_sd (&B[ temp + 2]);
				b4 = _mm256_broadcast_sd (&B[ temp + 3]);
				for (i = 0; i < (M - 3); i += 4) {
					c = _mm256_load_pd(&C[lda*j + i]);
					temp = lda*k+i;
					temp1 = _mm256_mul_pd(_mm256_load_pd(&A[temp]),b1);
					temp2 = _mm256_mul_pd(_mm256_load_pd(&A[temp += lda]), b2);
					temp3 = _mm256_mul_pd(_mm256_load_pd(&A[temp += lda]), b3);
					temp4 = _mm256_mul_pd(_mm256_load_pd(&A[temp += lda]), b4);

					temp5 = _mm256_add_pd(temp1, temp2);

					temp6 = _mm256_add_pd(temp3, temp4);

					c = _mm256_add_pd(c, temp5);

					c = _mm256_add_pd(c, temp6);

					_mm256_store_pd(&C[lda*j + i],c);

				}	
				if(M % 4){
					temp = j*lda+k;
					bb1 = B[temp];
					bb2 = B[temp+1];
					bb3 = B[temp+2];
					bb4 = B[temp+3];
					for (; i < M; ++i) {
						temp = lda*k+i;
						C[lda*j + i] += A[temp] * bb1;
						C[lda*j + i] += A[temp+=lda] * bb2;
						C[lda*j + i] += A[temp+=lda] * bb3;
						C[lda*j + i] += A[temp+=lda] * bb4;
					}
				}
			}
			if (K % 4) {
				do {
					b1 = _mm256_broadcast_sd (&B[ lda*j + k]);
					for (i = 0; i < (M - 3); i += 4) {
						c = _mm256_load_pd(&C[lda*j + i]);
						c = _mm256_add_pd(c, _mm256_mul_pd(_mm256_load_pd(&A[lda*k + i]), b1));
						_mm256_store_pd(&C[lda*j + i],c);
					}
					if(M % 4){
						bb1 = B[j*lda + k];
						for (; i < M; ++i) {				
							C[lda*j + i] += A[lda*k + i] * bb1;
						}
					}
				} while (++k < K);
			}
		} while (++j < N);
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
