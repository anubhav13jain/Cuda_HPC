#include<stdio.h>
#include<math.h>
#include<cuda.h>

__global__ void matMul(int *d_a,int *d_b,int *d_c, int M, int N, int K){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col= blockIdx.x * blockDim.x + threadIdx.x;

	if(row<M && col<K){
		int sum=d_a[row*N]*d_b[col];
		for(int k=1;k<N;k++){
			sum+=d_a[row*N+k]*d_b[k*K+col];
		}
		d_c[row*K + col]=sum;
	}
}

int main(int argc, char const *argv[]) {
	int N=8,M=10,K=9,TW=4;
	int h_a[M][N],h_b[N][K],h_c[M][K];
	int *d_a,*d_b,*d_c;

	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			h_a[i][j]=rand()%100;
		}
	}
	for(int i=0;i<N;i++){
		for(int j=0;j<K;j++){
			h_b[i][j]=rand()%100;
		}
	}

	printf("\n A Matrix\n" );
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			printf("%d ",h_a[i][j] );
		}
		printf("\n" );
	}
	printf("\n B Matrix\n" );
	for(int i=0;i<N;i++){
		for(int j=0;j<K;j++){
			printf("%d ",h_b[i][j] );
		}
		printf("\n" );
	}

	// taking block diamension as M X M
	dim3 dimBlock(TW,TW);
	dim3 dimGrid((int)((M-1)/TW)+1,(int)((K-1)/TW)+1,1 );

	// allocating device memory
	cudaMalloc(&d_a, M*N*sizeof(int));
	cudaMalloc(&d_b, N*K*sizeof(int));
	cudaMalloc(&d_c, M*K*sizeof(int));


	// copying data in device memory
	cudaMemcpy( d_a, h_a, M*N*sizeof(int), cudaMemcpyHostToDevice   );
	cudaMemcpy( d_b, h_b, N*K*sizeof(int), cudaMemcpyHostToDevice   );

	//calling kernel function

	matMul<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,M,N,K);

	cudaMemcpy(h_c, d_c, M*K*sizeof(int), cudaMemcpyDeviceToHost  );

	printf("\n Multiplication of A and B Matrix:\n" );
	for(int i=0;i<M;i++){
		for(int j=0;j<K;j++){
			printf("%d ",h_c[i][j] );
		}
		printf("\n" );
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
