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
#include <array>
#include <vector>
#include <stdlib.h>

// number of sites
#define N (2 * 1024 * 1024 / 4)

// number of sites
//~ #define DENSITY 0.8924
#define DENSITY 0.88

// number of temporal steps
#define NSTEPS 10000

using namespace std;

typedef double REAL;
typedef array<int,N> Manna_Array; // fixed-sized array (recien me entero de que esto existe en STL...)


// CONDICION INICIAL ---------------------------------------------------------------
/*
Para generar una condicion inicial suficientemente uniforme con una densidad
lo mas aproximada (exacta cuando N->infinito) al numero real DENSITY, podemos hacer asi:
*/
void inicializacion(Manna_Array &h)
{
	for(int i = 0; i < N; ++i) {
		h[i] = (int)((i+1)*DENSITY)-(int)(i*DENSITY);
	}
}

void imprimir_array(Manna_Array &h)
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


// CONDICION INICIAL ---------------------------------------------------------------
/*
El problema con la condicion inicial de arriba es que es estable, no tiene sitios activos
y por tanto no evolucionara. Hay que desestabilizarla de alguna forma.
Una forma es agarrar cada granito, y tirarlo a su izquierda o derecha aleatoriamente...
*/
void desestabilizacion_inicial(Manna_Array &h)
{
	vector<int> index_a_incrementar;
	for (int i = 0; i < N; ++i){
		if (h[i] == 1) {
			h[i] = 0;
			int j=i+2*(rand()%2)-1; // izquierda o derecha

			// corrijo por condiciones periodicas
			if (j == N) j = 0;
			if (j == -1) j = N-1;

			index_a_incrementar.push_back(j);
		}
	}
	for(unsigned int i = 0; i < index_a_incrementar.size(); ++i) {
		h[index_a_incrementar[i]] += 1;
	}
}

// DESCARGA DE ACTIVOS Y UPDATE --------------------------------------------------------
unsigned int descargar(Manna_Array &h, Manna_Array &dh)
{
	dh.fill(0);

	int i = 0;
	
	/* lo saco afuera del loop para simplificar el cálculo de k */
	// si es activo lo descargo aleatoriamente
	if (h[i] > 1) {
		for (int j = 0; j < h[i]; ++j) {
			// sitio receptor a la izquierda o derecha teniendo en cuenta condiciones periodicas
			int k = (i+2*(rand()&1)-1+N)%N; //&1 instead of %2
			++dh[k];
		}
		h[i] = 0;
	}
	
	for (i = 1; i < N-1; ++i) {
		// si es activo lo descargo aleatoriamente
		if (h[i] > 1) {
			for (int j = 0; j < h[i]; ++j) {
				// sitio receptor a la izquierda o derecha teniendo en cuenta condiciones periodicas
				//~ int k = (i+2*(rand()%2)-1+N)%N;
				int k = i+2*(rand()&1)-1; //&1 instead of %2
				++dh[k];
			}
			h[i] = 0;
		}
	}
	
	/* lo saco afuera del loop para simplificar el cálculo de k */
	// si es activo lo descargo aleatoriamente
	if (h[i] > 1) {
		for (int j = 0; j < h[i]; ++j) {
			// sitio receptor a la izquierda o derecha teniendo en cuenta condiciones periodicas
			int k = (i+2*(rand()&1)-1+N)%N; //&1 instead of %2
			++dh[k];
		}
		h[i] = 0;
	}

	unsigned int nroactivos=0;
	for (int i = 0; i < N; ++i) {
		h[i] += dh[i];
		nroactivos += (h[i]>1);
	}

	return nroactivos;
}

//===================================================================
// Lo compilo asi: g++ tiny_manna.cpp -std=c++0x
int main(){
	ios::sync_with_stdio(0); cin.tie(0);
	
	//~ srand(time(0));
	srand(12345);

	// nro granitos en cada sitio, y su update
	Manna_Array h, dh;

	cout << "estado inicial estable de la pila de arena...";
	inicializacion(h);
	cout << "LISTO\n";
	#ifdef DEBUG
	imprimir_array(h);
	#endif

	cout << "estado inicial desestabilizado de la pila de arena...";
	desestabilizacion_inicial(h);
	cout << "LISTO\n";
	#ifdef DEBUG
	imprimir_array(h);
	#endif

	cout << "evolucion de la pila de arena..."; cout.flush();

	ofstream activity_out("activity.dat");
	int activity;
	int t = 0;
	do {
		activity_out << (activity=descargar(h,dh)) << "\n";
		#ifdef DEBUG
		imprimir_array(h);
		#endif
		++t;
	} while(activity > 0 && t < NSTEPS); // si la actividad decae a cero, esto no evoluciona mas...

	cout << "LISTO: " << ((activity>0)?("se acabo el tiempo\n\n"):("la actividad decayo a cero\n\n")); cout.flush();

	return 0;
}
