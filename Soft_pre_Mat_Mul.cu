#include<stdio.h>
#include<math.h>
#include<cuda.h>
#define TWW 32 // Setting TileWidth

/*----------Kernel Function------------*/
__global__ void matMul_Soft_prefetch(double *d_a, double *d_b, double *d_c, int M, int N, int K){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col= blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ double ds_a[TWW][TWW];
	__shared__ double ds_b[TWW][TWW];

	double cval=0.0;
	
	// Loading data from Global meomory to register
	register double reg_1 =d_a[row*N+threadIdx.x];
	register double reg_2 =d_b[threadIdx.y*K+col];

	//For loopstarts from 1 (instead of 0 like in the tiled matrix multiplication) 
	//because we have already loaded data of phase 0 in register
	for(int t=1;t<N/TWW;t++){
		
		//Load data of current phase into shared memory 
		ds_a[threadIdx.y][threadIdx.x]=reg_1;
		ds_b[threadIdx.y][threadIdx.x]=reg_2;

		__syncthreads();
		
		//Starting fetching data for next phase in register
               	reg_1 = d_a[row*N+t*TWW+threadIdx.x];
		reg_2 = d_b[(t*TWW+threadIdx.y)*K+col];
		
		for(int k=0;k<TWW;k++){
			cval+=ds_a[threadIdx.y][k]*ds_b[k][threadIdx.x];
		}
		
		__syncthreads();

	}
	
	//For last phase data loading in shared memory
	ds_a[threadIdx.y][threadIdx.x]=reg_1;
	ds_b[threadIdx.y][threadIdx.x]=reg_2;

	__syncthreads();		
		
	for(int k=0;k<TWW;k++){
        	cval+=ds_a[threadIdx.y][k]*ds_b[k][threadIdx.x];
        }
	__syncthreads();

	d_c[row*K + col]=cval;

}
/*------------------------------*/

int main(int argc, char const *argv[]) {
	int N=800;
	int M=N,K=N;
	double h_a[M][N],h_b[N][K],h_c[M][K];
	double *d_a,*d_b,*d_c;
	cudaEvent_t start,stop;
	float ms;
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

	/*printf("\n A Matrix\n" );
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			printf("%0.1f ",h_a[i][j] );
		}
		printf("\n" );
	}
	printf("\n B Matrix\n" );
	for(int i=0;i<N;i++){
		for(int j=0;j<K;j++){
			printf("%0.1f ",h_b[i][j] );
		}
		printf("\n" );
	}*/

	// taking block diamension as TWW X TWW
	dim3 dimBlock(TWW,TWW);
	dim3 dimGrid(K/TWW,M/TWW );


	// allocating device memory
	cudaMalloc(&d_a, M*N*sizeof(double));
	cudaMalloc(&d_b, N*K*sizeof(double));
	cudaMalloc(&d_c, M*K*sizeof(double));


	// copying data in device memory
	cudaMemcpy( d_a, h_a, M*N*sizeof(double), cudaMemcpyHostToDevice   );
	cudaMemcpy( d_b, h_b, N*K*sizeof(double), cudaMemcpyHostToDevice   );

	//Creating timestamp event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//Recording Kernel start time
	cudaEventRecord(start, 0);	
	
	//calling kernel function
	matMul_Soft_prefetch<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,M,N,K);
	
	//Recording Kernel stop time
	cudaEventRecord(stop, 0);
	
	cudaMemcpy(h_c, d_c, M*K*sizeof(double), cudaMemcpyDeviceToHost  );

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	printf("\nTime:%f ", ms);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	/*printf("\n Multiplication of A and B Matrix using Tiling:\n" );
	for(int i=0;i<M;i++){
		for(int j=0;j<K;j++){
			printf("%0.1f ",h_c[i][j] );
		}
		printf("\n" );
	}*/
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}


