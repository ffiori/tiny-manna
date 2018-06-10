# Cómo testear con hugepages

## Preparación

La [wiki de Debian](https://wiki.debian.org/Hugepages) tiene buena información
introductoria. Asumiendo que nuestro hardware soporta hugepages de 2MB,
hay que activarlas y reservar algunas. Esto se hace agregando a la línea de
comandos del núcleo algo como lo siguiente, para reservar 100 (200M total):

    hugepagesz=2M hugepages=100

Luego al encender deberíamos verlas reservadas:

    $ hugeadm --pool-list
          Size  Minimum  Current  Maximum  Default
       2097152      100      100      100        *

También deshabilitaremos las transparent hugepages para que no impacten
en nuestras pruebas:

    # hugeadm --thp-never

## Pruebas

Para las pruebas se agrandó el tamaño N del programa, para que utilice más
memoria y tenga sentido evaluar hugepages. También se redujo la cantidad
de iteraciones para mantener el tiempo de ejecución razonable. Se modificó
en una de las copias la forma de manejar memoria para usar un allocator
con mmap y hugepages basado en un ejemplo del código fuente de Linux.


TODO: ver velocidad, parece que hugepages mejora un poco. Ver si es por
hugepages o por cambiar array<int, N> a punteros old school.
