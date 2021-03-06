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

#include <iostream>
#include <fstream>
#include <cstring>
#include <array>
#include <vector>
#include <cstdlib>
#include <random>
#include <cassert>

// number of sites
#ifdef NSLOTS
#define N NSLOTS
#else
#define N (1024*1024)
#endif

#define SIZE (N * 4)

#define BLOCK_SIZE 256

#define DENSITY 0.8924

// number of temporal steps
#define NSTEPS 10000

using namespace std;
typedef int * Manna_Array;

//fastest prng is XORWOW, default.
//~ #define curandState curandStatePhilox4_32_10_t 	//not so slow
//~ #define curandState curandStateMRG32k3a_t 		//slowest by far

__device__ curandState seed[1];
__device__ curandState rand_state[N];

__global__ void seedinit(int first_num){ //60ms, not top priority
	curand_init(first_num,0,0,seed);
	for(int i=0; i<N; i++) //must do it sequentially because of race conditions in curand(seed)
		curand_init(curand(seed),0,0,&rand_state[i]);
}

__device__ static inline bool randbool(curandState *rand_state){
	//~ return 1; //trick to fix behaviour
	return 1&curand(rand_state);
}

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

__device__ void imprimir_array(Manna_Array __restrict__ h)
{
	int nrogranitos=0;
	int nrogranitos_activos=0;

	// esto dibuja los granitos en cada sitio y los cuenta
	for(int i = 0; i < 10; ++i) {
		printf("%d ",h[i]);
		nrogranitos += h[i];
		nrogranitos_activos += (h[i]>1);
	}
	printf("\nHay %d granos en total\n",nrogranitos);
	printf("De ellos %d activos\n",nrogranitos_activos);
	printf("densidad obtenida es %f, mientras que la deseada era %f\n\n",nrogranitos*1.0/N,DENSITY);
}

__global__ void desestabilizacion_inicial(Manna_Array __restrict__ h, Manna_Array __restrict__ dh)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (h[gtid]) {
		int k = (gtid+2*randbool(&rand_state[gtid])-1+N)%N;
		//~ int k = (gtid+2*((gtid%3)%2)-1+N)%N; //trick to fix behavior
		atomicAdd(&dh[k], 1);
		h[gtid] = 0;
	}
}

__global__ void descargar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ slots_activos)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	//~ unsigned int tid = threadIdx.x; // id hilo dentro del bloque
	//~ unsigned int lane = tid & CUDA_WARP_MASK; // id hilo dentro del warp, aka lane
	//~ uint warp = tid / CUDA_WARP_SIZE;  // warp dentro del bloque
	//~ uint gwarp = gtid / CUDA_WARP_SIZE;  // Identificador global de warp
	//~ uint bid = blockIdx.x;  // Identificador de bloque
	
	curandState *thread_state = &rand_state[gtid]; //doesn't get better if I use a local copy and then copy back
	
	if (h[gtid] > 1) {
		for (int j = 0; j < h[gtid]; ++j) {
			int k = (gtid+2*randbool(thread_state)-1+N)%N;
			atomicAdd(&dh[k], 1);
		}
	} else atomicAdd(&dh[gtid], h[gtid]);
	h[gtid] = 0;

	if(gtid==0) *slots_activos=0;
}

__global__ void old_actualizar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	h[gtid]+=dh[gtid];
	dh[gtid]=0; 	//zeroes dh array
}

__global__ void actualizar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ slots_activos)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	if(h[gtid]>1)
		atomicAdd(slots_activos, 1);
}

__device__ unsigned int slots_activos;
/*
Dynamic parallelism:
When a parent thread block launches a child grid, the child is not guaranteed to begin
execution until the parent thread block reaches an explicit synchronization point (e.g.
cudaDeviceSynchronize()).
*/
__global__ void run_manna(unsigned int *activity, Manna_Array h, Manna_Array dh) {
	int t = 0;
	do {
		descargar<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,&activity[t]);
		Manna_Array tmp = h;
		h = dh;
		dh = tmp;
		actualizar<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,&activity[t]);

		//~ cudaDeviceSynchronize();
		//~ activity[t] = slots_activos;
		++t;
	//~ } while(slots_activos > 0 && t < NSTEPS); // si la actividad decae a cero, esto no evoluciona mas...
	} while(t < NSTEPS); // ahora corre todos los NSTEPS sí o sí
}

__device__ Manna_Array h,dh;
__device__ unsigned int *activity;
unsigned int activity_host[NSTEPS];

//===================================================================
int main(){
	ios::sync_with_stdio(0); cin.tie(0);
	assert(N%BLOCK_SIZE==0);
	
	//random initialization
	seedinit<<<1,1>>>(time(NULL)); //initialize a state per thread with some random seed
	getLastCudaError("seedinit failed");

	//slots
	checkCudaErrors(cudaMalloc(&h, N*sizeof(int)));
	checkCudaErrors(cudaMalloc(&dh, N*sizeof(int)));
	checkCudaErrors(cudaMalloc(&activity, NSTEPS*sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(dh, 0, N*sizeof(int)));

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
	desestabilizacion_inicial<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh);
	getLastCudaError("desestabilizacion failed");
	old_actualizar<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh);
	getLastCudaError("actualizar failed");
	cout << "LISTO\n";
	
	#ifdef DEBUG
	imprimir_array(h);
	#endif

	cout << "evolucion de la pila de arena..."; cout.flush();

	ofstream activity_out("activity.dat");

	run_manna<<<1,1>>>(activity,h,dh);
	checkCudaErrors(cudaMemcpy(activity_host, activity, sizeof(activity_host), cudaMemcpyDeviceToHost));

	bool timeout = true;
	for (int i = 0; i < NSTEPS; i++) {
		activity_out << activity_host[i] << "\n";
		if (!activity_host[i]) { timeout = false; break; }
	}

	cout << "LISTO: " << ((timeout)?("se acabo el tiempo\n\n"):("la actividad decayo a cero\n\n")); cout.flush();

	//free everything
	cudaFree(h);
	cudaFree(dh);
	cudaFree(activity);

	return 0;
}
