/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

#ifdef __CUDACC__
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];
#endif
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc, const float *pcCos, const float *pcSin)
{
  float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset(*acc, 0, sizeof(int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {
      int idx = j * w + i;
      if (pic[idx] > 0) {
        int xCoord = i - xCent;
        int yCoord = yCent - j;
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
          float r = xCoord * pcCos[tIdx] + yCoord * pcSin[tIdx];
          int rIdx = (int)((r + rMax) / rScale);
          if (rIdx >= 0 && rIdx < rBins) {
            (*acc)[rIdx * degreeBins + tIdx]++;
          }
        }
      }
    }
  }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;

  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      if (rIdx >= 0 && rIdx < rBins)
      {
        atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
      }
    }
  }
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
/*__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  // Cálculo del globalID: en una configuración 1D de bloques y threads
  // El globalID es el índice único de cada thread en el grid completo
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  // Conversión del índice global del pixel a coordenadas cartesianas centradas
  // El gloID es un índice lineal que representa la posición del pixel en la imagen
  // Si la imagen se almacena fila por fila: gloID = fila * ancho + columna
  // Entonces: columna = gloID % w, fila = gloID / w
  // xCoord: convierte la columna (0 a w-1) a coordenadas centradas (-w/2 a w/2)
  //         Esto es necesario porque la Transformada de Hough usa coordenadas
  //         cartesianas con origen en el centro de la imagen
  int xCoord = gloID % w - xCent;
  
  // yCoord: convierte la fila (0 a h-1) a coordenadas centradas (h/2 a -h/2)
  //         Se invierte (yCent - fila) porque en imágenes, el eje Y apunta hacia abajo
  //         (fila 0 está arriba), pero en coordenadas cartesianas Y apunta hacia arriba
  //         Esto permite que la Transformada de Hough funcione correctamente con el sistema
  //         de coordenadas cartesianas estándar
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          // Validar que rIdx esté en el rango correcto
          if (rIdx >= 0 && rIdx < rBins)
          {
            //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
          }
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}
*/
//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;
  
  printf("Dimensiones de la imagen: %d x %d\n", w, h);

  // Crear eventos CUDA para todas las mediciones (BITÁCORA - Versión Memoria Global)
  cudaEvent_t startTotal, stopTotal;
  cudaEvent_t startCPU, stopCPU;
  cudaEvent_t startPrecompute, stopPrecompute;
  cudaEvent_t startMallocGPU2, stopMallocGPU2;
  cudaEvent_t startMallocGPU3, stopMallocGPU3;
  cudaEvent_t startMemsetGPU, stopMemsetGPU;
  cudaEvent_t startH2D_CosSin, stopH2D_CosSin;
  cudaEvent_t startH2D_Image, stopH2D_Image;
  cudaEvent_t startKernel, stopKernel;
  cudaEvent_t startKernelSync, stopKernelSync;
  cudaEvent_t startD2H, stopD2H;
  cudaEvent_t startStats, stopStats;
  cudaEvent_t startOutputImg, stopOutputImg;
  
  // Crear todos los eventos
  cudaEventCreate(&startTotal);
  cudaEventCreate(&stopTotal);
  cudaEventCreate(&startCPU);
  cudaEventCreate(&stopCPU);
  cudaEventCreate(&startPrecompute);
  cudaEventCreate(&stopPrecompute);
  cudaEventCreate(&startMallocGPU2);
  cudaEventCreate(&stopMallocGPU2);
  cudaEventCreate(&startMallocGPU3);
  cudaEventCreate(&stopMallocGPU3);
  cudaEventCreate(&startMemsetGPU);
  cudaEventCreate(&stopMemsetGPU);
  cudaEventCreate(&startH2D_CosSin);
  cudaEventCreate(&stopH2D_CosSin);
  cudaEventCreate(&startH2D_Image);
  cudaEventCreate(&stopH2D_Image);
  cudaEventCreate(&startKernel);
  cudaEventCreate(&stopKernel);
  cudaEventCreate(&startKernelSync);
  cudaEventCreate(&stopKernelSync);
  cudaEventCreate(&startD2H);
  cudaEventCreate(&stopD2H);
  cudaEventCreate(&startStats);
  cudaEventCreate(&stopStats);
  cudaEventCreate(&startOutputImg);
  cudaEventCreate(&stopOutputImg);

  // Iniciar medición del tiempo total
  cudaEventRecord(startTotal);

 
 

  // 2. Medición: Tiempo de pre-cálculo de cos/sin
  cudaEventRecord(startPrecompute);
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cosf(rad);
    pcSin[i] = sinf(rad);
    rad += radInc;
  }
  cudaEventRecord(stopPrecompute);
  cudaEventSynchronize(stopPrecompute);

  cudaEventRecord(startCPU);
  CPU_HoughTran(inImg.pixels, w, h, &cpuht, pcCos, pcSin);
  cudaEventRecord(stopCPU);
  cudaEventSynchronize(stopCPU);

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // 4. Medición: Tiempo de transferencia Host to Device (cos/sin)
  cudaEventRecord(startH2D_CosSin);
  cudaError_t err1 = cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
  cudaError_t err2 = cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);
  if (err1 != cudaSuccess || err2 != cudaSuccess) {
      printf("Error al copiar a memoria constante: %s / %s\n",
          cudaGetErrorString(err1), cudaGetErrorString(err2));
  } else {
      printf("Copiados %d valores a memoria constante correctamente.\n", degreeBins);
  }
  cudaEventRecord(stopH2D_CosSin);
  cudaEventSynchronize(stopH2D_CosSin);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  // 5. Medición: Tiempo de alocación de memoria GPU (d_in)
  cudaEventRecord(startMallocGPU2);
  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaEventRecord(stopMallocGPU2);
  cudaEventSynchronize(stopMallocGPU2);
  
  // 6. Medición: Tiempo de alocación de memoria GPU (d_hough)
  cudaEventRecord(startMallocGPU3);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaEventRecord(stopMallocGPU3);
  cudaEventSynchronize(stopMallocGPU3);
  
  // 7. Medición: Tiempo de inicialización de memoria GPU (memset)
  cudaEventRecord(startMemsetGPU);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);
  cudaEventRecord(stopMemsetGPU);
  cudaEventSynchronize(stopMemsetGPU);
  
  // 8. Medición: Tiempo de transferencia Host to Device (imagen)
  cudaEventRecord(startH2D_Image);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaEventRecord(stopH2D_Image);
  cudaEventSynchronize(stopH2D_Image);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  
  int blockNum = ceil (1.0 * w * h / 256);
  
  // 9. Medición: Tiempo de ejecución del kernel GPU (memoria global)
  cudaEventRecord(startKernel);
  GPU_HoughTranConst <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  cudaError_t kerr = cudaGetLastError();
  if (kerr != cudaSuccess) {
      printf("Error lanzando kernel: %s\n", cudaGetErrorString(kerr));
  }
  cudaEventRecord(stopKernel);
  
  // 10. Medición: Tiempo de sincronización después del kernel
  cudaEventRecord(startKernelSync);
  cudaEventSynchronize(stopKernel);
  cudaEventRecord(stopKernelSync);
  cudaEventSynchronize(stopKernelSync);

  // 11. Medición: Tiempo de transferencia Device to Host
  cudaEventRecord(startD2H);
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);
  int sumGPU = 0;
  for (int i = 0; i < degreeBins * rBins; i++) sumGPU += h_hough[i];
  printf("Suma total acumulador GPU: %d\n", sumGPU);
  cudaEventRecord(stopD2H);
  cudaEventSynchronize(stopD2H);

  // Calcular todos los tiempos medidos (se inicializan aquí, se calculan después)
  float timeCPU = 0, timePrecompute = 0;
  float timeMallocGPU2 = 0, timeMallocGPU3 = 0;
  float timeMemsetGPU = 0, timeH2D_CosSin = 0, timeH2D_Image = 0;
  float timeKernel = 0, timeKernelSync = 0, timeD2H = 0;
  float timeStats = 0, timeOutputImg = 0;
  
  cudaEventElapsedTime(&timeCPU, startCPU, stopCPU);
  cudaEventElapsedTime(&timePrecompute, startPrecompute, stopPrecompute);
  cudaEventElapsedTime(&timeMallocGPU2, startMallocGPU2, stopMallocGPU2);
  cudaEventElapsedTime(&timeMallocGPU3, startMallocGPU3, stopMallocGPU3);
  cudaEventElapsedTime(&timeMemsetGPU, startMemsetGPU, stopMemsetGPU);
  cudaEventElapsedTime(&timeH2D_CosSin, startH2D_CosSin, stopH2D_CosSin);
  cudaEventElapsedTime(&timeH2D_Image, startH2D_Image, stopH2D_Image);
  cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);
  cudaEventElapsedTime(&timeKernelSync, startKernelSync, stopKernelSync);
  cudaEventElapsedTime(&timeD2H, startD2H, stopD2H);
  
  // Mostrar bitácora de tiempos
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════╗\n");
  printf("║   BITÁCORA DE TIEMPOS - KERNEL MEMORIA GLOBAL CONSTANTE    ║\n");
  printf("╠════════════════════════════════════════════════════════════╣\n");
  printf("║ 1. Tiempo ejecución CPU:              %10.3f ms ║\n", timeCPU);
  printf("║ 2. Tiempo pre-cálculo cos/sin:        %10.3f ms ║\n", timePrecompute);
  printf("║ 4. Tiempo transferencia H->D (cos/sin):%10.3f ms ║\n", timeH2D_CosSin);
  printf("║ 5. Tiempo alocación GPU (imagen):     %10.3f ms ║\n", timeMallocGPU2);
  printf("║ 6. Tiempo alocación GPU (acumulador):  %10.3f ms ║\n", timeMallocGPU3);
  printf("║ 7. Tiempo inicialización GPU (memset):%10.3f ms ║\n", timeMemsetGPU);
  printf("║ 8. Tiempo transferencia H->D (imagen):%10.3f ms ║\n", timeH2D_Image);
  printf("║ 9. Tiempo ejecución kernel GPU:        %10.3f ms ║\n", timeKernel);
  printf("║10. Tiempo sincronización kernel:      %10.3f ms ║\n", timeKernelSync);
  printf("║11. Tiempo transferencia D->H:        %10.3f ms ║\n", timeD2H);
  printf("╠════════════════════════════════════════════════════════════╣\n");
  float timeGPU_Total = timeMallocGPU2 + timeMallocGPU3 + 
                        timeMemsetGPU + timeH2D_CosSin + timeH2D_Image + 
                        timeKernel + timeKernelSync + timeD2H;
  printf("║ Tiempo total operaciones GPU:         %10.3f ms ║\n", timeGPU_Total);
  printf("╚════════════════════════════════════════════════════════════╝\n");
  printf("\n");

  // Diagnóstico: contar píxeles con valor > 0 en la imagen
  int pixelCount = 0;
  for (i = 0; i < w * h; i++)
  {
    if (inImg.pixels[i] > 0)
      pixelCount++;
  }
  printf("Píxeles con valor > 0 en la imagen: %d de %d (%.2f%%)\n", pixelCount, w * h, 100.0 * pixelCount / (w * h));

  // Diagnóstico: encontrar el máximo valor en el acumulador
  int maxVal = 0;
  int maxIdx = -1;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (h_hough[i] > maxVal)
    {
      maxVal = h_hough[i];
      maxIdx = i;
    }
  }
  printf("Valor máximo en el acumulador: %d (índice: %d)\n", maxVal, maxIdx);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");

  // Se implementó la generación de una imagen de salida (formato PPM) 
  //que muestra las líneas detectadas dibujadas en rojo sobre la imagen original 
  //en escala de grises. Solo se dibujan las líneas cuyo peso supera el threshold (promedio + 2 desviaciones estándar
  // 12. Medición: Tiempo de cálculo de estadísticas
  cudaEventRecord(startStats);
  long long sum = 0;
  long long sumSq = 0;
  int count = degreeBins * rBins;
  for (i = 0; i < count; i++)
  {
    sum += h_hough[i];
    sumSq += (long long)h_hough[i] * h_hough[i];
  }
  float mean = (float)sum / count;
  float variance = ((float)sumSq / count) - (mean * mean);
  float stddev = sqrt(variance);
  int threshold = (int)(mean + 2 * stddev);
  cudaEventRecord(stopStats);
  cudaEventSynchronize(stopStats);
  cudaEventElapsedTime(&timeStats, startStats, stopStats);
  printf("Pesos - Promedio: %.2f, Desviación estándar: %.2f, Threshold: %d\n", mean, stddev, threshold);
  printf("Tiempo cálculo estadísticas: %.3f ms\n", timeStats);

  // 13. Medición: Tiempo de generación de imagen de salida
  cudaEventRecord(startOutputImg);
  
  // Generar imagen de salida con líneas detectadas
  // Crear una imagen RGB (3 canales) para dibujar líneas a color
  unsigned char *outputImg = (unsigned char *)malloc(w * h * 3);
  
  // Copiar la imagen original en escala de grises a los 3 canales
  for (i = 0; i < w * h; i++)
  {
    outputImg[i * 3] = inImg.pixels[i];
    outputImg[i * 3 + 1] = inImg.pixels[i];
    outputImg[i * 3 + 2] = inImg.pixels[i];
  }

  // Dibujar líneas detectadas
  int xCent = w / 2;
  int yCent = h / 2;
  for (int rIdx = 0; rIdx < rBins; rIdx++)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      int weight = h_hough[rIdx * degreeBins + tIdx];
      if (weight > threshold)
      {
        // Convertir de (r, theta) a coordenadas cartesianas para dibujar la línea
        // El cálculo inverso: rIdx = (r + rMax) / rScale, entonces r = rIdx * rScale - rMax
        // Usamos el centro del bin para mayor precisión: r = (rIdx + 0.5) * rScale - rMax
        float r = (rIdx + 0.5f) * rScale - rMax;
        float theta = tIdx * radInc;
        float cosTheta = pcCos[tIdx];
        float sinTheta = pcSin[tIdx];
        
        // Dibujar la línea en la imagen
        // Usar la ecuación: x*cos(theta) + y*sin(theta) = r
        for (int x = 0; x < w; x++)
        {
          int xCoord = x - xCent;
          // Calcular y: y = (r - x*cos(theta)) / sin(theta)
          if (fabs(sinTheta) > 0.0001)
          {
            float yCoord = (r - xCoord * cosTheta) / sinTheta;
            int y = yCent - (int)yCoord; // Convertir de vuelta a coordenadas de imagen
            
            if (y >= 0 && y < h)
            {
              int idx = (y * w + x) * 3;
              // Dibujar línea en rojo
              outputImg[idx] = 255;     // R
              outputImg[idx + 1] = 0;    // G
              outputImg[idx + 2] = 0;    // B
            }
          }
        }
        
        // También dibujar por columnas para líneas verticales
        for (int y = 0; y < h; y++)
        {
          int yCoord = yCent - y;
          // Calcular x: x = (r - y*sin(theta)) / cos(theta)
          if (fabs(cosTheta) > 0.0001)
          {
            float xCoord = (r - yCoord * sinTheta) / cosTheta;
            int x = xCoord + xCent;
            
            if (x >= 0 && x < w)
            {
              int idx = (y * w + x) * 3;
              // Dibujar línea en rojo
              outputImg[idx] = 255;     // R
              outputImg[idx + 1] = 0;    // G
              outputImg[idx + 2] = 0;    // B
            }
          }
        }
      }
    }
  }

  // Guardar imagen de salida como PPM (formato simple, sin compresión)
  char outputFilename[256];
  // Extraer el nombre base del archivo sin extensión
  char baseName[256];
  strncpy(baseName, argv[1], sizeof(baseName) - 1);
  baseName[sizeof(baseName) - 1] = '\0';
  char *ext = strrchr(baseName, '.');
  if (ext) *ext = '\0';
  snprintf(outputFilename, sizeof(outputFilename), "%s_output.ppm", baseName);
  FILE *fout = fopen(outputFilename, "wb");
  if (fout)
  {
    fprintf(fout, "P6\n%d %d\n255\n", w, h);
    fwrite(outputImg, 1, w * h * 3, fout);
    fclose(fout);
    printf("Imagen de salida guardada como: %s\n", outputFilename);
  }
  else
  {
    printf("Error: No se pudo crear el archivo de salida\n");
  }
  
  cudaEventRecord(stopOutputImg);
  cudaEventSynchronize(stopOutputImg);
  cudaEventElapsedTime(&timeOutputImg, startOutputImg, stopOutputImg);
  printf("Tiempo generación imagen salida: %.3f ms\n", timeOutputImg);

  // Liberar memoria
  free(outputImg);
  cudaFree(d_in);
  cudaFree(d_hough);
  free(h_hough);
  free(pcCos);
  free(pcSin);
  delete[] cpuht;
  
  // Finalizar medición del tiempo total
  cudaEventRecord(stopTotal);
  cudaEventSynchronize(stopTotal);
  float timeTotal = 0;
  cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
  
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════╗\n");
  printf("║ TIEMPO TOTAL DEL PROGRAMA:            %10.3f ms ║\n", timeTotal);
  printf("╚════════════════════════════════════════════════════════════╝\n");
  printf("\n");
  
  // Destruir todos los eventos
  cudaEventDestroy(startTotal);
  cudaEventDestroy(stopTotal);
  cudaEventDestroy(startCPU);
  cudaEventDestroy(stopCPU);
  cudaEventDestroy(startPrecompute);
  cudaEventDestroy(stopPrecompute);
  cudaEventDestroy(startMallocGPU2);
  cudaEventDestroy(stopMallocGPU2);
  cudaEventDestroy(startMallocGPU3);
  cudaEventDestroy(stopMallocGPU3);
  cudaEventDestroy(startMemsetGPU);
  cudaEventDestroy(stopMemsetGPU);
  cudaEventDestroy(startH2D_CosSin);
  cudaEventDestroy(stopH2D_CosSin);
  cudaEventDestroy(startH2D_Image);
  cudaEventDestroy(stopH2D_Image);
  cudaEventDestroy(startKernel);
  cudaEventDestroy(stopKernel);
  cudaEventDestroy(startKernelSync);
  cudaEventDestroy(stopKernelSync);
  cudaEventDestroy(startD2H);
  cudaEventDestroy(stopD2H);
  cudaEventDestroy(startStats);
  cudaEventDestroy(stopStats);
  cudaEventDestroy(startOutputImg);
  cudaEventDestroy(stopOutputImg);

  return 0;
}
