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


#define L1_BLOCK_SIZE 256
#define L2_BLOCK_SIZE 512

#define min(a,b) (((a)<(b))?(a):(b))
#define COMPUTATION_ARR(A,i,j) (A)[(j)*lda + (i)]

static inline calc4_nt(int lda, int i, int j, int K, double* A, double* B, double* C){
	int temp;
	__m256d b1, b2, b3, b4, temp1, c1, c2, c3, c4, a1;
	c1 = _mm256_load_pd(&C[lda*j + i]);
	c2 = _mm256_load_pd(&C[lda*(j+1) + i]);
	c3 = _mm256_load_pd(&C[lda*(j+2) + i]);
	c4 = _mm256_load_pd(&C[lda*(j+3) + i]);

	for(int k = 0; k < K; ++k){
		temp = lda*j + k;
		b1 = _mm256_broadcast_sd (&B[ temp ]);
		b2 = _mm256_broadcast_sd (&B[ temp+lda ]);
		b3 = _mm256_broadcast_sd (&B[ temp+(2*lda)]);
		b4 = _mm256_broadcast_sd (&B[ temp+(3*lda)]);

		temp = lda*k+i;
		a1 = _mm256_load_pd( &A[temp]);

		c1 = _mm256_add_pd(c1, _mm256_mul_pd(a1,b1));
		c2 = _mm256_add_pd(c2, _mm256_mul_pd(a1,b2));
		c3 = _mm256_add_pd(c3, _mm256_mul_pd(a1,b3));
		c4 = _mm256_add_pd(c4, _mm256_mul_pd(a1,b4));

	}
	_mm256_store_pd(&C[lda*j + i],c1);
	_mm256_store_pd(&C[lda*(j+1) + i],c2);
	_mm256_store_pd(&C[lda*(j+2) + i],c3);
	_mm256_store_pd(&C[lda*(j+3) + i],c4);
}

static inline calc4_t(int lda, int i, int j, int K, double* a, double* b, double* c){
	int temp;
	__m256d b1, b2, b3, b4, temp1
	c1, c2, c3, c4, a1;
	double* cp2;
	double* cp3;
	double* cp4;
	cp2 = c + lda;
	cp3 = cp2 + lda;
	cp4 = cp3 + lda;


	c1 = _mm256_load_pd(c);
	c2 = _mm256_load_pd(cp2);
	c3 = _mm256_load_pd(cp3);
	c4 = _mm256_load_pd(cp4);

	for(int k = 0; k < K; ++k){

		b1 = _mm256_broadcast_sd (b++);
		b2 = _mm256_broadcast_sd (b++);
		b3 = _mm256_broadcast_sd (b++);
		b4 = _mm256_broadcast_sd (b++);

		a1 = _mm256_load_pd(a);
		a+= 4;

		c1 = _mm256_add_pd(c1, _mm256_mul_pd(a1,b1));
		c2 = _mm256_add_pd(c2, _mm256_mul_pd(a1,b2));
		c3 = _mm256_add_pd(c3, _mm256_mul_pd(a1,b3));
		c4 = _mm256_add_pd(c4, _mm256_mul_pd(a1,b4));

	}
	_mm256_store_pd(c,c1);
	_mm256_store_pd(cp2,c2);
	_mm256_store_pd(cp3,c3);
	_mm256_store_pd(cp4,c4);
}

 static inline void a_elements_copy (int lda, const int K, double* a_src, double* a_dest) {

  for (int i = 0; i < K; ++i)
  {
    *a_dest++ = *a_src;
    *a_dest++ = *(a_src+1);
    *a_dest++ = *(a_src+2);
    *a_dest++ = *(a_src+3);
    a_src += lda;
  }
}

static inline void b_elements_copy (int lda, const int K, double* b_src, double* b_dest) {
  double *pointer_b0, *pointer_b1, *pointer_b2, *pointer_b3;
  pointer_b0 = b_src;
  pointer_b1 = pointer_b0 + lda;
  pointer_b2 = pointer_b1 + lda;
  pointer_b3 = pointer_b2 + lda;

  for (int i = 0; i < K; ++i)
  {
    *b_dest++ = *pointer_b0++;
    *b_dest++ = *pointer_b1++;
    *b_dest++ = *pointer_b2++;
    *b_dest++ = *pointer_b3++;
  }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

 static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
 {
   double buff_a[M*K], buff_b[K*N];
   double *pointer_a, *pointer_b, *c;

   const int maximum_n = N-3;
   int maximum_m = M-3;
   int edge_case1 = M%4;
   int edge_case2 = N%4;

   int i = 0, j = 0, p = 0;


   for (j = 0 ; j < maximum_n; j += 4)
   {
     pointer_b = &buff_b[j*K];

     b_elements_copy(lda, K, B + j*lda, pointer_b);

     for (i = 0; i < maximum_m; i += 4) {
       pointer_a = &buff_a[i*K];
       if (j == 0) a_elements_copy(lda, K, A + i, pointer_a);
       c = C + i + j*lda;
      calc4_t(lda, K, pointer_a, pointer_b, c);
     }
   }

   if (edge_case1 != 0)
   {
     for ( ; i < M; ++i)
       for (p = 0; p < N; ++p)
       {
         double c_ip = COMPUTATION_ARR(C,i,p);
         for (int k = 0; k < K; ++k)
           c_ip += COMPUTATION_ARR(A,i,k) * COMPUTATION_ARR(B,k,p);
         COMPUTATION_ARR(C,i,p) = c_ip;
       }
   }
   if (edge_case2 != 0)
   {
     maximum_m = M - edge_case1;
     for ( ; j < N; ++j)
       for (i = 0; i < maximum_m; ++i)
       {
         double cij = COMPUTATION_ARR(C,i,j);
         for (int k = 0; k < K; ++k)
           cij += COMPUTATION_ARR(A,i,k) * COMPUTATION_ARR(B,k,j);
         COMPUTATION_ARR(C,i,j) = cij;
       }
   }
 }

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
 void square_dgemm (int lda, double* A, double* B, double* C)
 {
   for (int t = 0; t < lda; t += L2_BLOCK_SIZE)
   {
     int end_k = t + min(L2_BLOCK_SIZE, lda-t);

     for (int s = 0; s < lda; s += L2_BLOCK_SIZE)
     {
       int end_j = s + min(L2_BLOCK_SIZE, lda-s);

       for (int r = 0; r < lda; r += L2_BLOCK_SIZE)
       {
         int end_i = r + min(L2_BLOCK_SIZE, lda-r);
         for (int k = t; k < end_k; k += L1_BLOCK_SIZE)
         {
           int K = min(L1_BLOCK_SIZE, end_k-k);

           for (int j = s; j < end_j; j += L1_BLOCK_SIZE)
           {
             int N = min(L1_BLOCK_SIZE, end_j-j);

             for (int i = r; i < end_i; i += L1_BLOCK_SIZE)
             {
               int M = min(L1_BLOCK_SIZE, end_i-i);

               do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
             }
           }
         }
       }
     }
   }
 }
