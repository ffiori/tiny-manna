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
#define N (1024) //TODO: se rompe todo si compilás con -DN=123, cambiar de N a NSLOTS o algo así

#define SIZE (N * 4)

#define BLOCK_SIZE 32

#define DENSITY 0.8924

// number of temporal steps
#define NSTEPS 10000

using namespace std;
typedef int * Manna_Array;

#ifndef SEED
#define SEED 0
#endif

/*
void randinit() {
	random_device rd;
	generator = default_random_engine(SEED ? SEED : rd());
}

static inline bool randbool() {
	uniform_int_distribution<int> distribution(0,1);
	return distribution(generator);
}
*/

__device__ curandState_t *rand_state;

__global__ void randinit(curandState_t *rand_state){
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(gtid,gtid,0,&rand_state[gtid]); //TODO put cool seed, not gtid
}

__device__ bool randbool(){ //TODO needs fix
	return 1;
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	return 1&curand(&rand_state[gtid]);
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

__global__ void desestabilizacion_inicial(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ slots_activos)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (h[gtid]) {
		//~ int k = (gtid+2*randbool()-1+N)%N;
		int k = (gtid+2*((gtid%3)%2)-1+N)%N; //trick to fix behavior
		atomicAdd(&dh[k], 1);
		h[gtid] = 0;
	}
}

__global__ void descargar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ slots_activos)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	//~ unsigned int tid = threadIdx.x; // id hilo dentro del bloque
	//~ unsigned int lane = tid & CUDA_WARP_MASK; // id hilo dentro del warp, aka lane
	
	unsigned int i = gtid;

	// si es activo lo descargo aleatoriamente
	if (h[i] > 1) {
		for (int j = 0; j < h[i]; ++j) {
			int k = (i+2*randbool()-1+N)%N;
			atomicAdd(&dh[k], 1);
		}
		h[i] = 0;
	}
	 
	if(gtid==0) *slots_activos=0;
}

__global__ void actualizar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh, unsigned int * __restrict__ result)
{
	unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	h[gtid]+=dh[gtid];
	dh[gtid]=0; 	//zeroes dh array
	if(h[gtid]>1)
		atomicAdd(result, 1);
}

__device__ Manna_Array h,dh;
__device__ unsigned int slots_activos;

//===================================================================
int main(){
	ios::sync_with_stdio(0); cin.tie(0);
	assert(N%BLOCK_SIZE==0);
	
	//~ randinit();
	checkCudaErrors(cudaMalloc(&rand_state, N*sizeof(curandState_t)));
	randinit<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(rand_state);

	// nro granitos en cada sitio, y su update
	checkCudaErrors(cudaMalloc(&h, N*sizeof(int)));
	checkCudaErrors(cudaMalloc(&dh, N*sizeof(int)));

	//gets actual address in device (for example &slots_activos is garbage)
	unsigned int *slots_activos_addr;
	cudaGetSymbolAddress((void **)&slots_activos_addr, slots_activos);

	cout << "estado inicial estable de la pila de arena...";
	inicializacion<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(h);
	getLastCudaError("inicializacion failed");
	cout << "LISTO\n";
	
	#ifdef DEBUG
	imprimir_array(h);
	#endif

	cout << "estado inicial desestabilizado de la pila de arena...";
	desestabilizacion_inicial<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,slots_activos_addr);
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
		descargar<<< N/BLOCK_SIZE, BLOCK_SIZE >>>(h,dh,slots_activos_addr);
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

	return 0;
}
