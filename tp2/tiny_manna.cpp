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
#define printear256(leftold) _mm256_extract_epi32(leftold,0)<<" "<<_mm256_extract_epi32(leftold,1)<<" "<<_mm256_extract_epi32(leftold,2)<<" "<<_mm256_extract_epi32(leftold,3)<<" "<<_mm256_extract_epi32(leftold,4)<<" "<<_mm256_extract_epi32(leftold,5)<<" "<<_mm256_extract_epi32(leftold,6)<<" "<<_mm256_extract_epi32(leftold,7)

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

const __m256i zeroes = _mm256_set_epi32(0,0,0,0,0,0,0,0);
const __m128i zeroes128 = _mm_set_epi32(0,0,0,0);
const __m256i ones = _mm256_set_epi32(1,1,1,1,1,1,1,1);

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

static inline __m256i shift_half_right(__m256i input){
	return _mm256_set_m128i(_mm256_extracti128_si256(input,0), zeroes128);
}

static inline __m256i shift_half_left(__m256i input){
	return _mm256_set_m128i(zeroes128, _mm256_extracti128_si256(input,1));
}

static inline __m256i shift192left(__m256i input){
	return _mm256_set_epi64x(0,0,0,_mm256_extract_epi64(input,3));
}

static inline __m256i shift64right(__m256i input){
	return _mm256_set_epi64x(_mm256_extract_epi64(input,2), _mm256_extract_epi64(input,1), _mm256_extract_epi64(input,0), 0);
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

#ifdef DEBUG
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
#endif

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

#if 0
// DESCARGA DE ACTIVOS Y UPDATE --------------------------------------------------------
unsigned int descargar_two_loops(Manna_Array __restrict__ h_, Manna_Array __restrict__ dh_)
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
#endif

#define DHSZ 32

#ifdef DEBUG
void printdh(Manna_Array dh){
	cout<<"dh[]: ";
	for(int i=0; i<DHSZ; i++) cout<<dh[i]<<" ";
	cout<<endl;
}
#endif

#if 0
const __m128i zeroes = _mm_set_epi32(0,0,0,0);
const __m128i ones = _mm_set_epi32(1,1,1,1);

unsigned int descargar128(Manna_Array __restrict__ h, Manna_Array __restrict__ dh)
{
	//~ Manna_Array __restrict__ h = (Manna_Array) __builtin_assume_aligned(h_,127);
	//~ Manna_Array __restrict__ dh = (Manna_Array) __builtin_assume_aligned(dh_,127);
	
	unsigned int nroactivos = 0;
	memset(dh, 0, DHSZ*4);

	int i = 0;
	
	for (i = 0; i < 4; ++i) {
		if(h[i] > 1) {
			for (int j = 0; j < h[i]; ++j) {
				int k = (i+2*randbool()-1+DHSZ)%DHSZ;
				++dh[k];
			}
			h[i] = 0;
		}
		
		if(i>1){ //actualizo salvo h[0] y h[N-1]
			h[i-1] += dh[i-1];
			nroactivos += (h[i-1]>1);
		}
	}

	#ifdef DEBUG
	printdh(dh);
	#endif

	__m128i left = _mm_loadu_si128((__m128i *) &dh[i-1]); //i=4
	__m128i right = zeroes;

	for (; i < N-4; i+=4) {
		__m128i slots = _mm_load_si128((__m128i *) &h[i]);
		__m128i slots_gt1 = _mm_cmpgt_epi32(slots,ones); //slots greater than 1
		__m128i active_slots;
		
		bool activity = false;
		while(active_slots = _mm_and_si128(slots_gt1, _mm_cmpgt_epi32(slots,zeroes)), _mm_movemask_epi8(active_slots)){ //active_slots[0] or active_slots[1]
			activity = true;
			short unsigned r = rand(); //BEST
			__m128i randomright = _mm_set_epi32(r&1,(r>>1)&1,(r>>2)&1,(r>>3)&1); // BEST

			__m128i randomleft = _mm_xor_si128(randomright, ones);
			__m128i addright = _mm_and_si128(randomright, active_slots);
			__m128i addleft = _mm_and_si128(randomleft, active_slots);

			left = _mm_add_epi32(left, addleft);
			right = _mm_add_epi32(right, addright);

			slots = _mm_sub_epi32(slots, _mm_and_si128(active_slots, ones)); // slots - (active_slots & ones), le resto 1 a cada slot activo
		}

		//escribo en dh
		__m128i mitad = _mm_slli_si128(right,8); //OJO, invertido
		__m128i left_to_store = _mm_add_epi32(left,mitad);

		left = _mm_srli_si128(right,8); //OJO, invertido
		right = zeroes;
		
		if(activity) _mm_store_si128((__m128i *) &h[i],slots);
		
		//~ _mm_storeu_si128((__m128i *) &dh[i-1],left_to_store);
		
		//actualizo
		if(left_to_store[0] or left_to_store[1]){ //if (left_to_store != 0)
			slots = _mm_loadu_si128((__m128i *) &h[i-1]);
			slots = _mm_add_epi32(slots, left_to_store);
			_mm_storeu_si128((__m128i *) &h[i-1], slots);
		
		
			slots_gt1 = _mm_cmpgt_epi32(slots,ones); //slots greater than 1
			slots_gt1 = _mm_and_si128(slots_gt1,ones);
			
			//~ nroactivos += (slots_gt1[0]&1) + (slots_gt1[0]>>32) + (slots_gt1[1]&1) + (slots_gt1[1]>>32); //slower option
			
			slots_gt1 = _mm_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1, a2+a3, b0+b1, b2+b3 (pero a=b=slots_gt1)
			slots_gt1 = _mm_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1+a2+a3, b0+b1+b2+b3, a0+a1+a2+a3, b0+b1+b2+b3
			nroactivos += _mm_extract_epi32(slots_gt1,0);
		}
	}

	_mm_storeu_si128((__m128i *) &dh[(i-1)%DHSZ],left);

	#ifdef DEBUG
	printdh(dh);
	#endif

	for (i = N-4; i < N; ++i){
		if(h[i] > 1) {
			for (int j = 0; j < h[i]; ++j) {
				int k = (i+2*randbool()-1)%DHSZ;
				++dh[k];
			}
			h[i] = 0;
		}
		//actualizo
		h[i-1] += dh[(i-1)%DHSZ];
        nroactivos += (h[i-1]>1);
	}
	
	//actualizo N-1
    h[N-1] += dh[(N-1)%DHSZ];
    nroactivos += (h[N-1]>1);
    
	//actualizo 0
    h[0] += dh[0];
    nroactivos += (h[0]>1);

	#ifdef DEBUG
	printdh(dh);
	#endif

	return nroactivos;
}
#endif

#define NSIMD 8

unsigned int descargar(Manna_Array __restrict__ h, Manna_Array __restrict__ dh)
{
	unsigned int nroactivos = 0;
	memset(dh, 0, DHSZ*(sizeof(int)));

	int i = 0;
	
	for (i = 0; i < NSIMD; ++i) {
		if(h[i] > 1) {
			for (int j = 0; j < h[i]; ++j) {
				int k = (i+2*randbool()-1+DHSZ)%DHSZ;
				++dh[k];
			}
			h[i] = 0;
		}
		
		if(i>1){ //actualizo salvo h[0] y h[N-1]
			h[i-1] += dh[i-1];
			nroactivos += (h[i-1]>1);
		}
	}
	
	#ifdef DEBUG
	printdh(dh);
	#endif
	
	__m256i left = _mm256_loadu_si256((__m256i *) &dh[i-1]); //i=NSIMD
	__m256i right = zeroes;

	for (; i < N-NSIMD; i+=NSIMD) {
		__m256i slots = _mm256_load_si256((__m256i *) &h[i]);
		__m256i slots_gt1 = _mm256_cmpgt_epi32(slots,ones); //slots greater than 1
		__m256i active_slots; //va a tener 0xffff en el slot si está activo

		#ifdef DEBUG
		cout<<"\nSLOTS NOW: "<<printear256(slots)<<endl;
		#endif
		
		bool activity = false;
		while(active_slots = _mm256_and_si256(slots_gt1, _mm256_cmpgt_epi32(slots,zeroes)), _mm256_movemask_epi8(active_slots)){ //active_slots[0] or active_slots[1] or...
			activity = true;
			short unsigned r = rand(); //BEST
			__m256i randomright = _mm256_set_epi32(r,r>>1,r>>2,r>>3,r>>4,r>>5,r>>6,r>>7); // BEST
			randomright = _mm256_and_si256(randomright, ones);
			__m256i randomleft = _mm256_xor_si256(randomright, ones);
			
			#ifdef DEBUG
			cout<<"random left: "<<printear256(randomleft)<<". right: "<<printear256(randomright)<<endl;
			#endif
			
			__m256i addright = _mm256_and_si256(randomright, active_slots);
			__m256i addleft = _mm256_and_si256(randomleft, active_slots);

			left = _mm256_add_epi32(left, addleft);
			right = _mm256_add_epi32(right, addright);

			slots = _mm256_sub_epi32(slots, _mm256_and_si256(active_slots, ones)); // slots - (active_slots & ones), le resto 1 a cada slot activo
		}

		#ifdef DEBUG		
		cout<<"update left: "<<printear256(left)<<". right: "<<printear256(right)<<endl;
		#endif

		//escribo en dh
		__m256i solapado = shift64right(right); //valores sumados en right cuyos indices coinciden con indices sumados en left
		__m256i left_to_store = _mm256_add_epi32(left,solapado); //los junto para storearlos ahora

		#ifdef DEBUG
		cout<<"right vale "<<printear256(right)<<endl;
		cout<<"solapavale "<<printear256(solapado)<<endl;
		cout<<"leftS vale "<<printear256(left_to_store)<<endl;
		#endif

		left = shift192left(right); //acumulo los valores que no se solapan
		right = zeroes;
		
		if(activity) _mm256_store_si256((__m256i *) &h[i],slots);
		
		//~ _mm256_storeu_si256((__m256i *) &dh[i-1],left_to_store);
		
		//actualizo
		if(left_to_store[0] or left_to_store[1] or left_to_store[2] or left_to_store[3]){ //if (left_to_store != 0)
			slots = _mm256_loadu_si256((__m256i *) &h[i-1]);
			slots = _mm256_add_epi32(slots, left_to_store);
			_mm256_storeu_si256((__m256i *) &h[i-1], slots);
		
			#ifdef DEBUG
			cout<<"SLOTS tow: "<<printear256(slots)<<endl;
			#endif
		
			slots_gt1 = _mm256_cmpgt_epi32(slots,ones); //slots greater than 1
			slots_gt1 = _mm256_and_si256(slots_gt1,ones);
			
			//~ nroactivos += (slots_gt1[0]&1) + ((slots_gt1[0]>>32)&1) + (slots_gt1[1]&1) + ((slots_gt1[1]>>32)&1) + (slots_gt1[2]&1) + ((slots_gt1[2]>>32)&1) + (slots_gt1[3]&1) + ((slots_gt1[3]>>32)&1); //slower option
			
			slots_gt1 = _mm256_hadd_epi32(slots_gt1,shift_half_left(slots_gt1)); // = a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7 (pero a=b=slots_gt1) = a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7
			slots_gt1 = _mm256_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1+a2+a3, b0+b1+b2+b3, .....
			slots_gt1 = _mm256_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1+a2+a3 + b0+b1+b2+b3, ...
			nroactivos += _mm256_extract_epi32(slots_gt1,7);
		}
	}

	_mm256_storeu_si256((__m256i *) &dh[(i-1)%DHSZ],left);

	#ifdef DEBUG
	printdh(dh);
	#endif

	for (; i < N; ++i){
		if(h[i] > 1) {
			for (int j = 0; j < h[i]; ++j) {
				int k = (i+2*randbool()-1)%DHSZ;
				++dh[k];
			}
			h[i] = 0;
		}
		//actualizo
		h[i-1] += dh[(i-1)%DHSZ];
        nroactivos += (h[i-1]>1);
	}
	
	//actualizo N-1
    h[N-1] += dh[(N-1)%DHSZ];
    nroactivos += (h[N-1]>1);
    
	//actualizo 0
    h[0] += dh[0];
    nroactivos += (h[0]>1);

	#ifdef DEBUG
	printdh(dh);
	#endif

	return nroactivos;
}

//===================================================================
// Lo compilo asi: g++ tiny_manna.cpp -std=c++0x
int main(){
	ios::sync_with_stdio(0); cin.tie(0);

	randinit();
	
/*
int a[8]={1,2,3,4,5,6,7,8};
__m256i aver = _mm256_loadu_si256((__m256i *)a);
cout<<printear256(aver)<<endl;
__m256i aver1 = _mm256_set_m128i(_mm256_extracti128_si256(aver,0), zeroes128);
cout<<printear256(aver1)<<endl;
aver1 = _mm256_set_m128i(_mm256_extracti128_si256(aver,1), zeroes128);
cout<<printear256(aver1)<<endl;
aver1 = _mm256_set_m128i(zeroes128, _mm256_extracti128_si256(aver,0));
cout<<printear256(aver1)<<endl;
aver1 = _mm256_set_m128i(zeroes128, _mm256_extracti128_si256(aver,1));
cout<<printear256(aver1)<<endl;

_mm256_storeu_si256((__m256i *)a, aver1);
for(int i=0;i<8;i++)cout<<a[i]<<" ";
cout<<endl;

//~ 1 2 3 4 5 6 7 8
//~ 0 0 0 0 1 2 3 4
//~ 0 0 0 0 5 6 7 8
//~ 1 2 3 4 0 0 0 0
//~ 5 6 7 8 0 0 0 0
//~ 5 6 7 8 0 0 0 0 

aver1 = shift64right(aver);
cout<<printear256(aver1)<<endl;
aver1 = shift192left(aver);
cout<<printear256(aver1)<<endl;

return 0;
*/

	#ifdef DEBUG
	cout<<"maximo random: "<<RAND_MAX<<endl;
	#endif

	// nro granitos en cada sitio, y su update
	//~ Manna_Array h = (int*)alloc(SIZE), dh = (int*)alloc(SIZE);

	Manna_Array h = (Manna_Array) aligned_alloc(128, SIZE);
	Manna_Array dh = (Manna_Array) aligned_alloc(128, sizeof(int)*DHSZ);

	//alineo en -1 porque hay más loads/stores en -1. NO MEJORA
	//~ Manna_Array h = (Manna_Array) aligned_alloc(128, SIZE+4);
	//~ Manna_Array dh = (Manna_Array) aligned_alloc(128, sizeof(int)*DHSZ+4); //ver de alinear en -1 si conviene porque todos los store/load son así. Lo probé y no mejora
	//~ h++; dh++;

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
		//~ if(t==3) return 0;
		#endif
		++t;
	} while(activity > 0 && t < NSTEPS); // si la actividad decae a cero, esto no evoluciona mas...

	cout << "LISTO: " << ((activity>0)?("se acabo el tiempo\n\n"):("la actividad decayo a cero\n\n")); cout.flush();

	return 0;
}
