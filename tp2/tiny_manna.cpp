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
#include <x86intrin.h> //SIMD

#define printear(leftold) _mm_extract_epi32(leftold,0)<<" "<<_mm_extract_epi32(leftold,1)<<" "<<_mm_extract_epi32(leftold,2)<<" "<<_mm_extract_epi32(leftold,3)

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

static inline bool rand16() {
        uniform_int_distribution<int> distribution(0,(1<<4)-1);
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

const __m128i zeroes = _mm_set_epi32(0,0,0,0);
const __m128i ones = _mm_set_epi32(1,1,1,1);
	
// DESCARGA DE ACTIVOS Y UPDATE --------------------------------------------------------
unsigned int descargar(Manna_Array __restrict__ h_, Manna_Array __restrict__ dh_)
{
//h_[0] = 0x123456; //DUMMY

	Manna_Array __restrict__ h = (Manna_Array) __builtin_assume_aligned(h_,128);
	Manna_Array __restrict__ dh = (Manna_Array) __builtin_assume_aligned(dh_,128);

	memset(dh, 0, SIZE);

	int i = 0;

	//~ for (i = 0; i < N; ++i) {
		//~ // si es activo lo descargo aleatoriamente
		//~ if (h[i] > 1) {
		
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
			
			//~ h[i]+=0xfef011;
			/*
			int right=0;
			int limit=h[i];
			for (int j = 0; j < limit; ++j) right += randbool();
			dh[(i+1)%N] += right;
			dh[(i-1+N)%N] += h[i]-right;
			*/
			//~ for (int j = 0; j < h[i]; ++j) randbool() ? ++dh[(i+1)%N] : ++dh[(i-1+N)%N];
			
			//~ h[i]+=0xfef02;

			
			//original
			//~ for (int j = 0; j < h[i]; ++j) {
				//~ // sitio receptor a la izquierda o derecha teniendo en cuenta condiciones periodicas
				//~ int k = (i+2*randbool()-1+N)%N;
				//~ ++dh[k];
			//~ }
			//~ h[i] = 0;
		//~ }
	//~ }
	
	for (i = 0; i < 4; ++i) if(h[i] > 1) {
		for (int j = 0; j < h[i]; ++j) {
			int k = (i+2*randbool()-1+N)%N;
			++dh[k];
		}
		h[i] = 0;
	}

	//~ __m128i left = zeroes;
	__m128i left = _mm_loadu_si128((__m128i *) &dh[i-1]);
	__m128i right = zeroes;

	for (; i < N-4; i+=4) {
		__m128i slots = _mm_load_si128((__m128i *) &h[i]);
		const __m128i slots_gt1 = _mm_cmpgt_epi32(slots,ones); //slots greater than 1
		__m128i active_slots;
		
		bool activity = false;
		while(active_slots = _mm_and_si128(slots_gt1, _mm_cmpgt_epi32(slots,zeroes)), _mm_movemask_epi8(active_slots)){ //active_slots[0] or active_slots[1]
			activity = true;
			short unsigned r = rand(); //BEST
			//~ short unsigned r; _rdrand16_step(&r);

			__m128i randomright = _mm_set_epi32(r&1,(r>>1)&1,(r>>2)&1,(r>>3)&1); //TODO optimize // BEST
			
			//~ __m128i randomright = _mm_set_epi32(randbool(),randbool(),randbool(),randbool()); //TODO optimize
			
			//const int r = rand();
			//__m128i randomright = _mm_blend_epi32(ones,zeroes,r); //con un rand de 0 a (1<<4)-1 alcanza, se puede hacer con el de la STL
			
			__m128i randomleft = _mm_xor_si128(randomright, ones);
//~ #ifdef DEBUG
//~ cout<<"random left: "<<printear(randomleft)<<". right: "<<printear(randomright)<<endl;
//~ #endif
			__m128i addright = _mm_and_si128(randomright, active_slots);
			__m128i addleft = _mm_and_si128(randomleft, active_slots);

			left = _mm_add_epi32(left, addleft);
			right = _mm_add_epi32(right, addright);

			slots = _mm_sub_epi32(slots, _mm_and_si128(active_slots, ones)); // slots - (active_slots & ones), le resto 1 a cada slot activo
		}
//~ #ifdef DEBUG	
//~ cout<<"update l: "<<printear(left)<<" y r: "<<printear(right)<<endl;
//~ #endif

		//escribo en dh
		//~ __m128i mitad = _mm_srli_si128(right,8);
		__m128i mitad = _mm_slli_si128(right,8); //OJO, invertido
		left = _mm_add_epi32(left,mitad);
		
		//~ #ifdef DEBUG
		//~ cout<<"right vale "<<printear(right)<<endl;
		//~ cout<<"mitad vale "<<printear(mitad)<<endl;
		//~ cout<<"left  vale "<<printear(left)<<endl;
		//~ #endif
		
		
		//~ if(i<=4){ //TODO sacar afuera del loop la primera iteración donde i=4
			//~ __m128i leftold = _mm_loadu_si128((__m128i *) &dh[i-1]); // siempre en 0 salvo los bordes
			//~ left = _mm_add_epi32(left,leftold);
		//~ }

		_mm_storeu_si128((__m128i *) &dh[i-1],left);
		
		//~ mitad = _mm_and_si128(right,secondhalf); //_mm_set_epi32(0,0,_mm_extract_epi32(right,2),_mm_extract_epi32(right,3));
		//~ __m128i rightold = _mm_loadu_si128((__m128i *) &dh[i+1]);
		//~ right = _mm_add_epi32(rightold,mitad);
		//~ _mm_storeu_si128((__m128i *) &dh[i+1],right);
		
		//~ left = _mm_slli_si128(right,8);
		left = _mm_srli_si128(right,8); //OJO, invertido
		right = zeroes;
		
		//~ #ifdef DEBUG
		//~ cout<<"SLOTS habia "<<h[i]<<" "<<h[i+1]<<" "<<h[i+2]<<" "<<h[i+3]<<endl;
		//~ cout<<"ahora hay   "<<printear(slots)<<endl;
		//~ cout<<"y dh vale "<<dh[i-1]<<" "<<dh[i]<<" "<<dh[i+1]<<" "<<dh[i+2]<<endl;
		//~ #endif
		
		if(activity) _mm_store_si128((__m128i *) &h[i],slots);
	}

	//~ __m128i leftold = _mm_loadu_si128((__m128i *) &dh[i-1]);
	//~ left = _mm_add_epi32(left,leftold);
	_mm_storeu_si128((__m128i *) &dh[i-1],left);

	for (i = N-4; i < N; ++i) if(h[i] > 1) {
		for (int j = 0; j < h[i]; ++j) {
			int k = (i+2*randbool()-1+N)%N;
			++dh[k];
		}
		h[i] = 0;
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
/*
int a[8]={1,2,3,4,5,6,7,8};
__m128i aver = _mm_loadu_si128((__m128i *)a);
cout<<printear(aver)<<endl;
aver = _mm_srli_si128(aver,8);
cout<<printear(aver)<<endl;
return 0;
*/
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
