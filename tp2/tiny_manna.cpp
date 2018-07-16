/*
Jueguito:

1) Sea h[i] el numero de granitos en el sitio i, 0<i<N-1.

2) Si h[i]>1 el sitio i esta "activo".

3) Al tiempo t, un sitio "activo" se "descarga" completamente tirando cada uno de sus granitos aleatoriamente y con igual probabilidad a la izquierda o a la derecha (el numero total de granitos entonces se conserva).

4) Los sitios se descargan sincronicamente. Entonces, a tiempo (t+1), el sitio activo i tendra h[i]=0 solo si sus vecinos no le tiraron granitos a tiempo t.

5) Se define la actividad A como el numero de sitios activos, es decir el numero de sitios que quieren descargarse.
Notar que si la densidad de granitos, [Suma_i h[i]/N] es muy baja, la actividad caera rapidamente a cero. Si la densidad es alta por otro lado, la actividad nunca cesara, ya que siempre habra sitios activos. En el medio hay una densidad "critica", para la cual la actividad decaera como una ley de potencia (pero se necesitaran sistemas grandes, y tiempos largos para verla bien definida).

*/

#include <iostream>
#include <fstream>
#include <cstring>
#include <array>
#include <vector>
#include <cstdlib>
#include <random>

#include <malloc.h>
//#include "huge-alloc.h"

#define INTSZ 32

// number of sites
//#define N (1024 / 4) //2MB data

#define SIZE (N * 4)

//~ #define DENSITY 0.8924
#define DENSITY 0.88

// number of temporal steps
#define NSTEPS 10000

using namespace std;

typedef double REAL;
typedef int * Manna_Array;

static default_random_engine generator;

#ifndef SEED
#define SEED 0
#endif

void randinit() {
	random_device rd;
	generator = default_random_engine(SEED ? SEED : rd());
	srand(SEED ? SEED : time(NULL));
}

static inline bool randbool() {
        uniform_int_distribution<int> distribution(0,1);
        return distribution(generator);
}


// CONDICION INICIAL ---------------------------------------------------------------
/*
Para generar una condicion inicial suficientemente uniforme con una densidad
lo mas aproximada (exacta cuando N->infinito) al numero real DENSITY, podemos hacer asi:
*/
void inicializacion(Manna_Array __restrict__ h)
{
	for(int i = 0; i < N; ++i) {
		h[i] = (int)((i+1)*DENSITY)-(int)(i*DENSITY);
	}
}

void imprimir_array(Manna_Array __restrict__ h)
{
	int nrogranitos=0;
	int nrogranitos_activos=0;

	// esto dibuja los granitos en cada sitio y los cuenta
	for(int i = 0; i < N; ++i) {
		//~ if(h[i]>5)
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


// CONDICION INICIAL ---------------------------------------------------------------
/*
El problema con la condicion inicial de arriba es que es estable, no tiene sitios activos
y por tanto no evolucionara. Hay que desestabilizarla de alguna forma.
Una forma es agarrar cada granito, y tirarlo a su izquierda o derecha aleatoriamente...
*/
void desestabilizacion_inicial(Manna_Array __restrict__ h)
{
	vector<int> index_a_incrementar;
	for (int i = 0; i < N; ++i){
		if (h[i] == 1) {
			h[i] = 0;
			int j=(i+2*randbool()-1+N)%N; // izquierda o derecha

			// corrijo por condiciones periodicas
			//if (j == N) j = 0;
			//if (j == -1) j = N-1;

			index_a_incrementar.push_back(j);
		}
	}
	for(unsigned int i = 0; i < index_a_incrementar.size(); ++i) {
		h[index_a_incrementar[i]] += 1;
	}
}

// DESCARGA DE ACTIVOS Y UPDATE --------------------------------------------------------
unsigned int descargar(Manna_Array __restrict__ h_, Manna_Array __restrict__ dh_)
{
//h_[0] = 0x123456; //DUMMY

	Manna_Array __restrict__ h = (Manna_Array) __builtin_assume_aligned(h_,128);
	Manna_Array __restrict__ dh = (Manna_Array) __builtin_assume_aligned(dh_,128);

	memset(dh, 0, SIZE);

	int i = 0;

	for (i = 0; i < N; ++i) {
		// si es activo lo descargo aleatoriamente
		if (h[i] > 1) {
			/*
			unsigned int r = rand() << (INTSZ-h[i]);
			int right = __builtin_popcount(r);
			
			dh[(i+1)%N] += right;
			dh[(i-1+N)%N] += h[i]-right;
			*/
			
			//~ while(h[i]){
				//~ int qty = min(h[i],32);
				//~ h[i]-=qty;
				//~ uniform_int_distribution<int> distribution(0,(1LL<<qty)-1);
				//~ int right = __builtin_popcount(distribution(generator));
				//~ //cout<<qty<<" "<<(1LL<<qty)-1<<" "<<distribution(generator)<<" "<<right<<endl;
				//~ dh[(i+1)%N] += right;
				//~ dh[(i-1+N)%N] += qty-right;
			//~ }
			
			//int right=0;
			//for (int j = 0; j < h[i]; ++j) right += randbool();
			//dh[(i+1)%N] += right;
			//dh[(i-1+N)%N] += h[i]-right;
			
			for (int j = 0; j < h[i]; ++j) {
				//~ // sitio receptor a la izquierda o derecha teniendo en cuenta condiciones periodicas
				int k = (i+2*randbool()-1+N)%N;
				++dh[k];
			}
			
			h[i] = 0;
		}
	}

//h[0] = 0x7777; //DUMMY

	unsigned int nroactivos=0;
	for (int i = 0; i < N; ++i) {
		h[i] += dh[i];
		nroactivos += (h[i]>1);
	}

//~ h[0] = 0xFEF0; //DUMMY xD

	return nroactivos;
}

//===================================================================
// Lo compilo asi: g++ tiny_manna.cpp -std=c++0x
int main(){
	ios::sync_with_stdio(0); cin.tie(0);

	randinit();

	#ifdef DEBUG
	cout<<"maximo random: "<<RAND_MAX<<endl;
	#endif

	// nro granitos en cada sitio, y su update
	//~ Manna_Array h = (int*)alloc(SIZE), dh = (int*)alloc(SIZE);
	Manna_Array h = (Manna_Array) aligned_alloc(128, SIZE), dh = (Manna_Array) aligned_alloc(128, SIZE);

	cout << "estado inicial estable de la pila de arena...";
	inicializacion(h);
	cout << "LISTO\n";
	#ifdef DEBUG
	//~ imprimir_array(h);
	#endif

	cout << "estado inicial desestabilizado de la pila de arena...";
	desestabilizacion_inicial(h);
	cout << "LISTO\n";
	#ifdef DEBUG
	//~ imprimir_array(h);
	#endif

	cout << "evolucion de la pila de arena..."; cout.flush();

	ofstream activity_out("activity.dat");
	int activity;
	int t = 0;
	do {
		activity_out << (activity=descargar(h,dh)) << "\n";
		#ifdef DEBUG
		//~ if(t and t%100==0)
			imprimir_array(h);
		#endif
		++t;
	} while(activity > 0 && t < NSTEPS); // si la actividad decae a cero, esto no evoluciona mas...

	cout << "LISTO: " << ((activity>0)?("se acabo el tiempo\n\n"):("la actividad decayo a cero\n\n")); cout.flush();

	return 0;
}
