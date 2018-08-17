/*
Jueguito:

1) Sea h[i] el numero de granitos en el sitio i, 0<i<N-1.

2) Si h[i]>1 el sitio i esta "activo".

3) Al tiempo t, un sitio "activo" se "descarga" completamente tirando cada uno de sus granitos aleatoriamente y con igual probabilidad a la izquierda o a la derecha (el numero total de granitos entonces se conserva).

4) Los sitios se descargan sincronicamente. Entonces, a tiempo (t+1), el sitio activo i tendra h[i]=0 solo si sus vecinos no le tiraron granitos a tiempo t.

5) Se define la actividad A como el numero de sitios activos, es decir el numero de sitios que quieren descargarse.
Notar que si la densidad de granitos, [Suma_i h[i]/N] es muy baja, la actividad caera rapidamente a cero. Si la densidad es alta por otro lado, la actividad nunca cesara, ya que siempre habra sitios activos. En el medio hay una densidad "critica", para la cual la actividad decaera como una ley de potencia (pero se necesitaran sistemas grandes, y tiempos largos para verla bien definida).

*/

#include <cuda.h>
#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "curand.h"
#include "curand_kernel.h"

#include "warp_rng.cuh"

#include <iostream>
#include <fstream>
#include <cstring>
#include <array>
#include <vector>
#include <cstdlib>
#include <random>
#include <cassert>

// number of sites
#define N (1024*1024) //TODO: se rompe todo si compilás con -DN=123, cambiar de N a NSLOTS o algo así
#define SIZE (N * 4)

#define BLOCK_SIZE 256
#define RAND_OFFSET 1

#define DENSITY 0.8924

// number of temporal steps
#define NSTEPS 10000

using namespace std;
typedef int * Manna_Array;

//fastest prng is XORWOW, default.
//~ #define curandState curandStatePhilox4_32_10_t 	//not so slow
//~ #define curandState curandStateMRG32k3a_t 		//slowest by far

__device__ curandState seed[1];
__device__ curandState *rand_state;
__device__ unsigned rand_state_warp[N];

__global__ void seedinit(int first_num, curandState *rand_state){ //120ms, not top priority
	curand_init(first_num,0,0,seed);
	for(int i=1; i<N; i++){ //must do it sequentially because of race conditions in curand(seed)
		curand_init(curand(seed),0,0,rand_state+i*RAND_OFFSET);
		rand_state_warp[i]=curand(seed);
	}
	rand_state_warp[0]=curand(seed);
}

__device__ static inline bool randbool(curandState *rand_state){
	//~ return 1; //trick to fix behaviour
	return 1&curand(rand_state); //TODO optimize perhaps?
}

/*
__device__ static inline bool randbool(curandState *rand_state) //sloweeeeeer
{
	static int random;
	static int calls=0;
	if(calls++%32==0) random=curand(rand_state);
	else random>>=1;
	return random&1;
}
*/

// CONDICION INICIAL ---------------------------------------------------------------
/*
Para generar una condicion inicial suficientemente uniforme con una densidad
lo mas aproximada (exacta cuando N->infinito) al numero real DENSITY, podemos hacer asi:
*/
__global__ void inicializacion(Manna_Array __restrict__ h)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	h[i] = (int)((i+1)*DENSITY)-(int)(i*DENSITY);
}

#ifdef DEBUG
void imprimir_array(Manna_Array __restrict__ h)
{
	int nrogranitos=0;
	int nrogranitos_activos=0;

	// esto dibuja los granitos en cada sitio y los cuenta
	for(int i = 0; i < N; ++i) {
		cout << h[i] << " ";
		nrogranitos += h[i];
		nrogranitos_activos += (h[i]>1);
	}
	cout << "\n";
	cout << "Hay " << nrogranitos << " granitos en total\n";
	cout << "De ellos " << nrogranitos_activos << " son activos\n";
	cout << "La densidad obtenida es " << nrogranitos*1.0/N;
	cout << ", mientras que la deseada era " << DENSITY << "\n\n";
}
#endif

__global__ void desestabilizacion_inicial(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ slots_activos, curandState *rand_state)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	curandState *thread_state = &rand_state[gtid*RAND_OFFSET];
	
	if (h[gtid]) {
		int k = (gtid+2*randbool(thread_state)-1+N)%N;
		//~ int k = (gtid+2*((gtid%3)%2)-1+N)%N; //trick to fix behavior
		atomicAdd(&dh[k], 1);
		h[gtid] = 0;
	}
}

extern __shared__ unsigned sharedMemory[BLOCK_SIZE];

__global__ void descargar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ slots_activos, curandState *rand_state)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	//~ unsigned int tid = threadIdx.x; // id hilo dentro del bloque
	//~ unsigned int lane = tid & CUDA_WARP_MASK; // id hilo dentro del warp, aka lane
	//~ uint warp = tid / CUDA_WARP_SIZE;  // warp dentro del bloque
	//~ uint gwarp = gtid / CUDA_WARP_SIZE;  // Identificador global de warp
	//~ uint bid = blockIdx.x;  // Identificador de bloque
	
	//~ curandState *thread_state = &rand_state[gtid*RAND_OFFSET]; //doesn't get better if I use a local copy and then copy back
	
	//~ // si es activo lo descargo aleatoriamente
	//~ if (h[gtid] > 1) {
		//~ for (int j = 0; j < h[gtid]; ++j) {
			//~ int k = (gtid+2*randbool(thread_state)-1+N)%N;
			//~ atomicAdd(&dh[k], 1);
		//~ }
		//~ h[gtid] = 0;
	//~ }


	// This should point to blockDim.x words of shared memory used to hold the RNG state.
	unsigned *rngShmem=sharedMemory;

	// These are private words for the RNG
	unsigned rngRegs[WarpCorrelated_REG_COUNT];

	// Move the state into the actual RNG
	WarpCorrelated_LoadState(rand_state_warp, rngRegs, sharedMemory);

	if (h[gtid] > 1) {
		for (int j = 0; j < h[gtid]; ++j) {
			int k = (gtid+2*(1&WarpCorrelated_Generate(rngRegs, sharedMemory))-1+N)%N;
			atomicAdd(&dh[k], 1);
		}
		h[gtid] = 0;
	}
	
	WarpCorrelated_SaveState(rngRegs, rngShmem, rand_state_warp);
	
/*
	int granos = h[gtid];
	if (granos > 1) {
		int r = (gtid+1)%N;
		int l = (gtid-1+N)%N;
		
		//~ float granos = h[gtid];
		//~ float random_value;
		//~ do{
			//~ random_value = curand_normal(thread_state)*granos/4.0 + granos/2.0;
		//~ }while(random_value>=granos+0.5 or random_value<=-0.5);
		//~ int left = roundf(random_value);
		
		//~ int left = 0;
		//~ do{
			//~ int random_value = curand(thread_state);
			//~ left += __popc(random_value);
			//~ granos-=32;
		//~ }while(granos>0);
		//~ int right = h[gtid]-left;

		
		//~ if(left<0 or left>h[gtid]){
			//~ printf("random: %f debe estar entre -0.5 y %f. left %d, right %d\n",random_value,granos+0.5,left,right);
			//~ assert(left>=0);
			//~ assert(left<=h[gtid]);
		//~ }
		
		atomicAdd(&dh[l], left);
		atomicAdd(&dh[r], right);
		
		h[gtid] = 0;
	}
*/	
	
	if(gtid==0) *slots_activos=0;
}

__global__ void actualizar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ result)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	//~ unsigned int tid = threadIdx.x; // id hilo dentro del bloque
	//~ unsigned int lane = tid & CUDA_WARP_MASK; // id hilo dentro del warp, aka lane
	
	assert(gtid<N);
	
	h[gtid]+=dh[gtid];
	dh[gtid]=0; 	//zeroes dh array
	if(h[gtid]>1) atomicAdd(result, 1);
}

__device__ Manna_Array dh,h;
__device__ unsigned int slots_activos;

//===================================================================
int main(){
	ios::sync_with_stdio(0); cin.tie(0);
	assert(N%BLOCK_SIZE==0);
	
	//random initialization
	checkCudaErrors(cudaMalloc(&rand_state, RAND_OFFSET*N*sizeof(curandState)));
	seedinit<<<1,1>>>(time(NULL),rand_state); //initialize a state per thread with some random seed
	getLastCudaError("seedinit failed");

	//slots
	checkCudaErrors(cudaMalloc(&h, N*sizeof(int)));
	checkCudaErrors(cudaMalloc(&dh, N*sizeof(int)));
	checkCudaErrors(cudaMemset(dh, 0, N*sizeof(int)));

	//gets actual address in device (&slots_activos is garbage)
	unsigned int *slots_activos_addr;
	cudaGetSymbolAddress((void **)&slots_activos_addr, slots_activos);

	//initialize slots
	cout << "estado inicial estable de la pila de arena...";
	inicializacion<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(h);
	getLastCudaError("inicializacion failed");
	cout << "LISTO\n";
	
	#ifdef DEBUG
	imprimir_array(h);
	#endif

	//create some chaos among slots
	cout << "estado inicial desestabilizado de la pila de arena...";
	desestabilizacion_inicial<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,slots_activos_addr,rand_state);
	getLastCudaError("desestabilizacion failed");
	actualizar<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,slots_activos_addr);
	getLastCudaError("actualizar failed");
	cout << "LISTO\n";
	
	#ifdef DEBUG
	imprimir_array(h);
	#endif

	cout << "evolucion de la pila de arena..."; cout.flush();

	ofstream activity_out("activity.dat");
	unsigned int activity;
	int t = 0;
	do {
		descargar<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,slots_activos_addr,rand_state);
		getLastCudaError("descargar failed");
		actualizar<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,slots_activos_addr);
		getLastCudaError("actualizar failed");
		checkCudaErrors(cudaMemcpyFromSymbol(&activity, slots_activos, sizeof(unsigned int)));
		
		activity_out << activity << "\n";
		#ifdef DEBUG
		imprimir_array(h);
		#endif
		++t;
	} while(activity > 0 && t < NSTEPS); // si la actividad decae a cero, esto no evoluciona mas...

	cout << "LISTO: " << ((activity>0)?("se acabo el tiempo\n\n"):("la actividad decayo a cero\n\n")); cout.flush();

	//free everything
	cudaFree(h);
	cudaFree(dh);
	cudaFree(rand_state);
	cudaFree(seed);

	return 0;
}

/*
 * TODO:
 * 		Try more work per thread. Change algorithm to get rid of many atomicAdd
 * 		make N and BLOCK_SIZE defineable during compile time
 * 		try normal distribution with: int curand_discrete(curandState_t *state, curandDiscreteDistribution_t discrete_distribution)
 */
