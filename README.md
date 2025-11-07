
## 1. Paralelización con Memoria Global

### 1.1 Compilación y Configuración

Se ajustó el **Makefile** para compilar correctamente el proyecto:

```makefile
all: pgm.o	hough

hough:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough -std=c++11

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o -std=c++11

clean:
	rm -f pgm.o hough
```

## 2. Implementación Completa de los Programas

### 2.1 Compilación
```makefile
make clean
make
```

### 2.2 Ejecución de Versión Base (Global + Constante)
```makefile
./hough runway.pgm
```

### 2.3 Ejeción de Versión con Memoria Compartida (Shared Memory)
```makefile
./hough runway.pgm shared
```

## 3. Benchmarking
El programa incluye un modo especial de benchmarking que ejecuta 10 corridas de cada versión y calcula automáticamente:
- promedio del tiempo del kernel
- promedio del tiempo total en GPU
- desviación estándar
- tiempo total del programa

### 3.1 Cómo ejecutar el benchmark
```makefile
./hough benchmark
```
> El programa mostrará primero 10 corridas del modo Global + Constante, y luego 10 corridas del modo Shared + Constante
> - Resultados promedio
> - Desviaciones estándar
> - Comparación entre ambos métodos

## 4. Referencias

- NVIDIA Developer Blog: [How to Implement Performance Metrics in CUDA C/C++](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)
- CUDA Programming Guide: CUDA Events for Performance Measurement
- Hough Transform: Detección de líneas en visión por computadora



