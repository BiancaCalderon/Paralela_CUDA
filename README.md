
## 1.## Paralelización con Memoria Global

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


## 4. Referencias

- NVIDIA Developer Blog: [How to Implement Performance Metrics in CUDA C/C++](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)
- CUDA Programming Guide: CUDA Events for Performance Measurement
- Hough Transform: Detección de líneas en visión por computadora


