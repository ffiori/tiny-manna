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

#include <cassert>
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

//~ #define zeroes (_mm256_setzero_si256())
const __m256i zeroes = _mm256_setzero_si256();
const __m128i zeroes128 = _mm_setzero_si128();
const __m256i ones = _mm256_set1_epi32(1); //broadcasts 1

#ifndef SEED
#define SEED 0
#endif

void randinit() {
	random_device rd;
	generator = default_random_engine(SEED ? SEED : rd());
	//~ srand(SEED ? SEED : time(NULL));
}

static inline bool randbool() {
	uniform_int_distribution<int> distribution(0,1);
	return distribution(generator);
}

static inline unsigned char randchar() {
	uniform_int_distribution<unsigned char> distribution(0,255);
	return distribution(generator);
}

static inline __m256i shift_half_right(__m256i input){ //not used
	return _mm256_set_m128i(_mm256_extracti128_si256(input,0), zeroes128);
}

static inline __m256i shift_half_left(__m256i input){ //not used
	return _mm256_set_m128i(zeroes128, _mm256_extracti128_si256(input,1));
}

const __m256i maskfff0 = _mm256_set_epi64x(-1,-1,-1,0);
const __m256i mask000f = _mm256_set_epi64x(0,0,0,-1);

static inline __m256i shift192left(__m256i input){
	return _mm256_and_si256(mask000f, _mm256_permute4x64_epi64(input,_MM_SHUFFLE(2,1,0,3)));
}

static inline __m256i shift64right(__m256i input){
	return _mm256_and_si256(maskfff0, _mm256_permute4x64_epi64(input,_MM_SHUFFLE(2,1,0,3)));
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
			index_a_incrementar.push_back(j);
		}
	}
	for(unsigned int i = 0; i < index_a_incrementar.size(); ++i) {
		h[index_a_incrementar[i]] += 1;
	}
}

#define DHSZ 32

#ifdef DEBUG
void printdh(Manna_Array dh){
	cout<<"dh[]: ";
	for(int i=0; i<DHSZ; i++) cout<<dh[i]<<" ";
	cout<<endl;
}
#endif

#define NSIMD 8

const __m256i MASK[256] = {
_mm256_blend_epi32(ones,zeroes, 0 ), _mm256_blend_epi32(ones,zeroes, 1 ), _mm256_blend_epi32(ones,zeroes, 2 ), _mm256_blend_epi32(ones,zeroes, 3 ), _mm256_blend_epi32(ones,zeroes, 4 ), _mm256_blend_epi32(ones,zeroes, 5 ), _mm256_blend_epi32(ones,zeroes, 6 ), _mm256_blend_epi32(ones,zeroes, 7 ), _mm256_blend_epi32(ones,zeroes, 8 ), _mm256_blend_epi32(ones,zeroes, 9 ), _mm256_blend_epi32(ones,zeroes, 10 ), _mm256_blend_epi32(ones,zeroes, 11 ), _mm256_blend_epi32(ones,zeroes, 12 ), _mm256_blend_epi32(ones,zeroes, 13 ), _mm256_blend_epi32(ones,zeroes, 14 ), _mm256_blend_epi32(ones,zeroes, 15 ), _mm256_blend_epi32(ones,zeroes, 16 ), _mm256_blend_epi32(ones,zeroes, 17 ), _mm256_blend_epi32(ones,zeroes, 18 ), _mm256_blend_epi32(ones,zeroes, 19 ), _mm256_blend_epi32(ones,zeroes, 20 ), _mm256_blend_epi32(ones,zeroes, 21 ), _mm256_blend_epi32(ones,zeroes, 22 ), _mm256_blend_epi32(ones,zeroes, 23 ), _mm256_blend_epi32(ones,zeroes, 24 ), _mm256_blend_epi32(ones,zeroes, 25 ), _mm256_blend_epi32(ones,zeroes, 26 ), _mm256_blend_epi32(ones,zeroes, 27 ), _mm256_blend_epi32(ones,zeroes, 28 ), _mm256_blend_epi32(ones,zeroes, 29 ), _mm256_blend_epi32(ones,zeroes, 30 ), _mm256_blend_epi32(ones,zeroes, 31 ), _mm256_blend_epi32(ones,zeroes, 32 ), _mm256_blend_epi32(ones,zeroes, 33 ), _mm256_blend_epi32(ones,zeroes, 34 ), _mm256_blend_epi32(ones,zeroes, 35 ), _mm256_blend_epi32(ones,zeroes, 36 ), _mm256_blend_epi32(ones,zeroes, 37 ), _mm256_blend_epi32(ones,zeroes, 38 ), _mm256_blend_epi32(ones,zeroes, 39 ), _mm256_blend_epi32(ones,zeroes, 40 ), _mm256_blend_epi32(ones,zeroes, 41 ), _mm256_blend_epi32(ones,zeroes, 42 ), _mm256_blend_epi32(ones,zeroes, 43 ), _mm256_blend_epi32(ones,zeroes, 44 ), _mm256_blend_epi32(ones,zeroes, 45 ), _mm256_blend_epi32(ones,zeroes, 46 ), _mm256_blend_epi32(ones,zeroes, 47 ), _mm256_blend_epi32(ones,zeroes, 48 ), _mm256_blend_epi32(ones,zeroes, 49 ), _mm256_blend_epi32(ones,zeroes, 50 ), _mm256_blend_epi32(ones,zeroes, 51 ), _mm256_blend_epi32(ones,zeroes, 52 ), _mm256_blend_epi32(ones,zeroes, 53 ), _mm256_blend_epi32(ones,zeroes, 54 ), _mm256_blend_epi32(ones,zeroes, 55 ), _mm256_blend_epi32(ones,zeroes, 56 ), _mm256_blend_epi32(ones,zeroes, 57 ), _mm256_blend_epi32(ones,zeroes, 58 ), _mm256_blend_epi32(ones,zeroes, 59 ), _mm256_blend_epi32(ones,zeroes, 60 ), _mm256_blend_epi32(ones,zeroes, 61 ), _mm256_blend_epi32(ones,zeroes, 62 ), _mm256_blend_epi32(ones,zeroes, 63 ), _mm256_blend_epi32(ones,zeroes, 64 ), _mm256_blend_epi32(ones,zeroes, 65 ), _mm256_blend_epi32(ones,zeroes, 66 ), _mm256_blend_epi32(ones,zeroes, 67 ), _mm256_blend_epi32(ones,zeroes, 68 ), _mm256_blend_epi32(ones,zeroes, 69 ), _mm256_blend_epi32(ones,zeroes, 70 ), _mm256_blend_epi32(ones,zeroes, 71 ), _mm256_blend_epi32(ones,zeroes, 72 ), _mm256_blend_epi32(ones,zeroes, 73 ), _mm256_blend_epi32(ones,zeroes, 74 ), _mm256_blend_epi32(ones,zeroes, 75 ), _mm256_blend_epi32(ones,zeroes, 76 ), _mm256_blend_epi32(ones,zeroes, 77 ), _mm256_blend_epi32(ones,zeroes, 78 ), _mm256_blend_epi32(ones,zeroes, 79 ), _mm256_blend_epi32(ones,zeroes, 80 ), _mm256_blend_epi32(ones,zeroes, 81 ), _mm256_blend_epi32(ones,zeroes, 82 ), _mm256_blend_epi32(ones,zeroes, 83 ), _mm256_blend_epi32(ones,zeroes, 84 ), _mm256_blend_epi32(ones,zeroes, 85 ), _mm256_blend_epi32(ones,zeroes, 86 ), _mm256_blend_epi32(ones,zeroes, 87 ), _mm256_blend_epi32(ones,zeroes, 88 ), _mm256_blend_epi32(ones,zeroes, 89 ), _mm256_blend_epi32(ones,zeroes, 90 ), _mm256_blend_epi32(ones,zeroes, 91 ), _mm256_blend_epi32(ones,zeroes, 92 ), _mm256_blend_epi32(ones,zeroes, 93 ), _mm256_blend_epi32(ones,zeroes, 94 ), _mm256_blend_epi32(ones,zeroes, 95 ), _mm256_blend_epi32(ones,zeroes, 96 ), _mm256_blend_epi32(ones,zeroes, 97 ), _mm256_blend_epi32(ones,zeroes, 98 ), _mm256_blend_epi32(ones,zeroes, 99 ), _mm256_blend_epi32(ones,zeroes, 100 ), _mm256_blend_epi32(ones,zeroes, 101 ), _mm256_blend_epi32(ones,zeroes, 102 ), _mm256_blend_epi32(ones,zeroes, 103 ), _mm256_blend_epi32(ones,zeroes, 104 ), _mm256_blend_epi32(ones,zeroes, 105 ), _mm256_blend_epi32(ones,zeroes, 106 ), _mm256_blend_epi32(ones,zeroes, 107 ), _mm256_blend_epi32(ones,zeroes, 108 ), _mm256_blend_epi32(ones,zeroes, 109 ), _mm256_blend_epi32(ones,zeroes, 110 ), _mm256_blend_epi32(ones,zeroes, 111 ), _mm256_blend_epi32(ones,zeroes, 112 ), _mm256_blend_epi32(ones,zeroes, 113 ), _mm256_blend_epi32(ones,zeroes, 114 ), _mm256_blend_epi32(ones,zeroes, 115 ), _mm256_blend_epi32(ones,zeroes, 116 ), _mm256_blend_epi32(ones,zeroes, 117 ), _mm256_blend_epi32(ones,zeroes, 118 ), _mm256_blend_epi32(ones,zeroes, 119 ), _mm256_blend_epi32(ones,zeroes, 120 ), _mm256_blend_epi32(ones,zeroes, 121 ), _mm256_blend_epi32(ones,zeroes, 122 ), _mm256_blend_epi32(ones,zeroes, 123 ), _mm256_blend_epi32(ones,zeroes, 124 ), _mm256_blend_epi32(ones,zeroes, 125 ), _mm256_blend_epi32(ones,zeroes, 126 ), _mm256_blend_epi32(ones,zeroes, 127 ), _mm256_blend_epi32(ones,zeroes, 128 ), _mm256_blend_epi32(ones,zeroes, 129 ), _mm256_blend_epi32(ones,zeroes, 130 ), _mm256_blend_epi32(ones,zeroes, 131 ), _mm256_blend_epi32(ones,zeroes, 132 ), _mm256_blend_epi32(ones,zeroes, 133 ), _mm256_blend_epi32(ones,zeroes, 134 ), _mm256_blend_epi32(ones,zeroes, 135 ), _mm256_blend_epi32(ones,zeroes, 136 ), _mm256_blend_epi32(ones,zeroes, 137 ), _mm256_blend_epi32(ones,zeroes, 138 ), _mm256_blend_epi32(ones,zeroes, 139 ), _mm256_blend_epi32(ones,zeroes, 140 ), _mm256_blend_epi32(ones,zeroes, 141 ), _mm256_blend_epi32(ones,zeroes, 142 ), _mm256_blend_epi32(ones,zeroes, 143 ), _mm256_blend_epi32(ones,zeroes, 144 ), _mm256_blend_epi32(ones,zeroes, 145 ), _mm256_blend_epi32(ones,zeroes, 146 ), _mm256_blend_epi32(ones,zeroes, 147 ), _mm256_blend_epi32(ones,zeroes, 148 ), _mm256_blend_epi32(ones,zeroes, 149 ), _mm256_blend_epi32(ones,zeroes, 150 ), _mm256_blend_epi32(ones,zeroes, 151 ), _mm256_blend_epi32(ones,zeroes, 152 ), _mm256_blend_epi32(ones,zeroes, 153 ), _mm256_blend_epi32(ones,zeroes, 154 ), _mm256_blend_epi32(ones,zeroes, 155 ), _mm256_blend_epi32(ones,zeroes, 156 ), _mm256_blend_epi32(ones,zeroes, 157 ), _mm256_blend_epi32(ones,zeroes, 158 ), _mm256_blend_epi32(ones,zeroes, 159 ), _mm256_blend_epi32(ones,zeroes, 160 ), _mm256_blend_epi32(ones,zeroes, 161 ), _mm256_blend_epi32(ones,zeroes, 162 ), _mm256_blend_epi32(ones,zeroes, 163 ), _mm256_blend_epi32(ones,zeroes, 164 ), _mm256_blend_epi32(ones,zeroes, 165 ), _mm256_blend_epi32(ones,zeroes, 166 ), _mm256_blend_epi32(ones,zeroes, 167 ), _mm256_blend_epi32(ones,zeroes, 168 ), _mm256_blend_epi32(ones,zeroes, 169 ), _mm256_blend_epi32(ones,zeroes, 170 ), _mm256_blend_epi32(ones,zeroes, 171 ), _mm256_blend_epi32(ones,zeroes, 172 ), _mm256_blend_epi32(ones,zeroes, 173 ), _mm256_blend_epi32(ones,zeroes, 174 ), _mm256_blend_epi32(ones,zeroes, 175 ), _mm256_blend_epi32(ones,zeroes, 176 ), _mm256_blend_epi32(ones,zeroes, 177 ), _mm256_blend_epi32(ones,zeroes, 178 ), _mm256_blend_epi32(ones,zeroes, 179 ), _mm256_blend_epi32(ones,zeroes, 180 ), _mm256_blend_epi32(ones,zeroes, 181 ), _mm256_blend_epi32(ones,zeroes, 182 ), _mm256_blend_epi32(ones,zeroes, 183 ), _mm256_blend_epi32(ones,zeroes, 184 ), _mm256_blend_epi32(ones,zeroes, 185 ), _mm256_blend_epi32(ones,zeroes, 186 ), _mm256_blend_epi32(ones,zeroes, 187 ), _mm256_blend_epi32(ones,zeroes, 188 ), _mm256_blend_epi32(ones,zeroes, 189 ), _mm256_blend_epi32(ones,zeroes, 190 ), _mm256_blend_epi32(ones,zeroes, 191 ), _mm256_blend_epi32(ones,zeroes, 192 ), _mm256_blend_epi32(ones,zeroes, 193 ), _mm256_blend_epi32(ones,zeroes, 194 ), _mm256_blend_epi32(ones,zeroes, 195 ), _mm256_blend_epi32(ones,zeroes, 196 ), _mm256_blend_epi32(ones,zeroes, 197 ), _mm256_blend_epi32(ones,zeroes, 198 ), _mm256_blend_epi32(ones,zeroes, 199 ), _mm256_blend_epi32(ones,zeroes, 200 ), _mm256_blend_epi32(ones,zeroes, 201 ), _mm256_blend_epi32(ones,zeroes, 202 ), _mm256_blend_epi32(ones,zeroes, 203 ), _mm256_blend_epi32(ones,zeroes, 204 ), _mm256_blend_epi32(ones,zeroes, 205 ), _mm256_blend_epi32(ones,zeroes, 206 ), _mm256_blend_epi32(ones,zeroes, 207 ), _mm256_blend_epi32(ones,zeroes, 208 ), _mm256_blend_epi32(ones,zeroes, 209 ), _mm256_blend_epi32(ones,zeroes, 210 ), _mm256_blend_epi32(ones,zeroes, 211 ), _mm256_blend_epi32(ones,zeroes, 212 ), _mm256_blend_epi32(ones,zeroes, 213 ), _mm256_blend_epi32(ones,zeroes, 214 ), _mm256_blend_epi32(ones,zeroes, 215 ), _mm256_blend_epi32(ones,zeroes, 216 ), _mm256_blend_epi32(ones,zeroes, 217 ), _mm256_blend_epi32(ones,zeroes, 218 ), _mm256_blend_epi32(ones,zeroes, 219 ), _mm256_blend_epi32(ones,zeroes, 220 ), _mm256_blend_epi32(ones,zeroes, 221 ), _mm256_blend_epi32(ones,zeroes, 222 ), _mm256_blend_epi32(ones,zeroes, 223 ), _mm256_blend_epi32(ones,zeroes, 224 ), _mm256_blend_epi32(ones,zeroes, 225 ), _mm256_blend_epi32(ones,zeroes, 226 ), _mm256_blend_epi32(ones,zeroes, 227 ), _mm256_blend_epi32(ones,zeroes, 228 ), _mm256_blend_epi32(ones,zeroes, 229 ), _mm256_blend_epi32(ones,zeroes, 230 ), _mm256_blend_epi32(ones,zeroes, 231 ), _mm256_blend_epi32(ones,zeroes, 232 ), _mm256_blend_epi32(ones,zeroes, 233 ), _mm256_blend_epi32(ones,zeroes, 234 ), _mm256_blend_epi32(ones,zeroes, 235 ), _mm256_blend_epi32(ones,zeroes, 236 ), _mm256_blend_epi32(ones,zeroes, 237 ), _mm256_blend_epi32(ones,zeroes, 238 ), _mm256_blend_epi32(ones,zeroes, 239 ), _mm256_blend_epi32(ones,zeroes, 240 ), _mm256_blend_epi32(ones,zeroes, 241 ), _mm256_blend_epi32(ones,zeroes, 242 ), _mm256_blend_epi32(ones,zeroes, 243 ), _mm256_blend_epi32(ones,zeroes, 244 ), _mm256_blend_epi32(ones,zeroes, 245 ), _mm256_blend_epi32(ones,zeroes, 246 ), _mm256_blend_epi32(ones,zeroes, 247 ), _mm256_blend_epi32(ones,zeroes, 248 ), _mm256_blend_epi32(ones,zeroes, 249 ), _mm256_blend_epi32(ones,zeroes, 250 ), _mm256_blend_epi32(ones,zeroes, 251 ), _mm256_blend_epi32(ones,zeroes, 252 ), _mm256_blend_epi32(ones,zeroes, 253 ), _mm256_blend_epi32(ones,zeroes, 254 ), _mm256_blend_epi32(ones,zeroes, 255 )
};

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
			unsigned char r = randchar();
			__m256i randomright = MASK[r];
			
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
		cout<<"solapavale "<<printear256(solapado)<<endl;
		cout<<"leftS vale "<<printear256(left_to_store)<<endl;
		#endif

		left = shift192left(right); //acumulo los valores que no se solapan
		right = zeroes;
		
		if(activity) _mm256_store_si256((__m256i *) &h[i],slots);
		
		//actualizo
		if(!_mm256_testz_si256(left_to_store,left_to_store)){ //if (left_to_store != 0)
			slots = _mm256_loadu_si256((__m256i *) &h[i-1]);
			slots = _mm256_add_epi32(slots, left_to_store);
			_mm256_storeu_si256((__m256i *) &h[i-1], slots);
		
			#ifdef DEBUG
			cout<<"SLOTS tow: "<<printear256(slots)<<endl;
			#endif
		
			__m256i tmp = _mm256_cmpgt_epi32(slots,ones); //slots greater than 1
			nroactivos += __builtin_popcount(_mm256_movemask_epi8(tmp))/4;
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
//~ __m256i aver1 = _mm256_set_m128i(_mm256_extracti128_si256(aver,0), zeroes128);
//~ cout<<printear256(aver1)<<endl;
//~ aver1 = _mm256_set_m128i(_mm256_extracti128_si256(aver,1), zeroes128);
//~ cout<<printear256(aver1)<<endl;
//~ aver1 = _mm256_set_m128i(zeroes128, _mm256_extracti128_si256(aver,0));
//~ cout<<printear256(aver1)<<endl;
//~ aver1 = _mm256_set_m128i(zeroes128, _mm256_extracti128_si256(aver,1));
//~ cout<<printear256(aver1)<<endl;

//~ _mm256_storeu_si256((__m256i *)a, aver1);
//~ for(int i=0;i<8;i++)cout<<a[i]<<" ";
//~ cout<<endl;

//~ 1 2 3 4 5 6 7 8
//~ 0 0 0 0 1 2 3 4
//~ 0 0 0 0 5 6 7 8
//~ 1 2 3 4 0 0 0 0
//~ 5 6 7 8 0 0 0 0
//~ 5 6 7 8 0 0 0 0 

//~ __m256i slots_gt1 = _mm256_hadd_epi32(aver,shift_half_left(aver)); // = a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7 (pero a=b=slots_gt1) = a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7
//~ slots_gt1 = _mm256_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1+a2+a3, b0+b1+b2+b3, .....
//~ slots_gt1 = _mm256_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1+a2+a3 + b0+b1+b2+b3, ...
//~ cout<<printear256(slots_gt1)<<endl;

//~ slots_gt1 = aver; // = a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7 (pero a=b=slots_gt1) = a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7
//~ slots_gt1 = _mm256_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1+a2+a3, b0+b1+b2+b3, .....
//~ slots_gt1 = _mm256_hadd_epi32(slots_gt1,slots_gt1); // = a0+a1+a2+a3 + b0+b1+b2+b3, ...
//~ cout<<printear256(slots_gt1)<<endl;
//~ cout<<_mm256_extract_epi32(slots_gt1,0) + _mm256_extract_epi32(slots_gt1,7)<<endl;

__m256i input = aver;
cout<<printear256(input)<<endl;
input = _mm256_permute4x64_epi64(aver, (3<<6)+(2<<4)+(1<<2));
cout<<printear256(input)<<endl;
input = _mm256_permute4x64_epi64(aver, (1<<6)+(2<<4)+(3<<2));
cout<<printear256(input)<<endl;
input = _mm256_permute4x64_epi64(aver, (3<<4)+(2<<2)+1);
cout<<printear256(input)<<endl;
input = _mm256_permute4x64_epi64(aver, (1<<4)+(2<<2)+3);
cout<<printear256(input)<<endl;

input = _mm256_permute4x64_epi64(aver, (2<<6) + (1<<4));
input = _mm256_permute4x64_epi64(aver,_MM_SHUFFLE(2,1,0,3));
cout<<printear256(input)<<endl;

input = shift192left(aver);
cout<<printear256(input)<<endl;
input = shift64right(aver);
cout<<printear256(input)<<endl;
return 0;
*/

	#ifdef DEBUG
	cout<<"maximo random: "<<RAND_MAX<<endl;
	#endif

	// nro granitos en cada sitio, y su update
	//~ Manna_Array h = (int*)alloc(SIZE), dh = (int*)alloc(SIZE);

	Manna_Array h = (Manna_Array) aligned_alloc(128, SIZE);
	Manna_Array dh = (Manna_Array) aligned_alloc(128, sizeof(int)*DHSZ);

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
