#include "pgm.h"
#include <fstream>
#include <iostream>
#include <sstream>

PGMImage::PGMImage(const char *filename) {
  readPGM(filename);
}

PGMImage::~PGMImage() {
  if (pixels != NULL) {
    delete[] pixels;
    pixels = NULL;
  }
}

void PGMImage::readPGM(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    fprintf(stderr, "Error: No se pudo abrir el archivo %s\n", filename);
    exit(1);
  }

  char magic[10];
  if (fgets(magic, sizeof(magic), fp) == NULL) {
    fprintf(stderr, "Error: Archivo PGM inválido (no se pudo leer el magic number)\n");
    fclose(fp);
    exit(1);
  }

  // Verificar que es un archivo PGM (P5 = binario, P2 = ASCII)
  if (magic[0] != 'P' || (magic[1] != '5' && magic[1] != '2')) {
    fprintf(stderr, "Error: El archivo no es un PGM válido (magic number leído: %s)\n", magic);
    fclose(fp);
    exit(1);
  }
  
  bool isBinary = (magic[1] == '5');

  // Leer dimensiones
  int maxVal = 255;
  char line[256];
  
  // Saltar comentarios
  do {
    if (fgets(line, sizeof(line), fp) == NULL) {
      fprintf(stderr, "Error: Archivo PGM inválido\n");
      fclose(fp);
      exit(1);
    }
  } while (line[0] == '#');

  // Leer ancho y alto
  int items = sscanf(line, "%d %d", &x_dim, &y_dim);
  if (items != 2 || x_dim <= 0 || y_dim <= 0) {
    fprintf(stderr, "Error: No se pudieron leer las dimensiones correctamente (x_dim=%d, y_dim=%d)\n", x_dim, y_dim);
    fprintf(stderr, "Línea leída: %s", line);
    fclose(fp);
    exit(1);
  }
  
  // Leer valor máximo
  if (fgets(line, sizeof(line), fp) == NULL) {
    fprintf(stderr, "Error: Archivo PGM inválido (no se pudo leer el valor máximo)\n");
    fclose(fp);
    exit(1);
  }
  items = sscanf(line, "%d", &maxVal);
  if (items != 1) {
    fprintf(stderr, "Error: No se pudo leer el valor máximo\n");
    fclose(fp);
    exit(1);
  }
  
  // Debug: mostrar información de la imagen
  fprintf(stderr, "Imagen cargada: %dx%d, maxVal=%d, formato=%s\n", x_dim, y_dim, maxVal, isBinary ? "P5 (binario)" : "P2 (ASCII)");

  // Asignar memoria para los pixeles
  pixels = new unsigned char[x_dim * y_dim];
  
  if (isBinary) {
    // Formato binario P5
    size_t bytesRead = fread(pixels, 1, x_dim * y_dim, fp);
    if (bytesRead != (size_t)(x_dim * y_dim)) {
      fprintf(stderr, "Error: No se pudieron leer todos los pixeles (leídos: %zu, esperados: %d)\n", 
              bytesRead, x_dim * y_dim);
      delete[] pixels;
      fclose(fp);
      exit(1);
    }
  } else {
    // Formato ASCII P2
    int value;
    for (int i = 0; i < x_dim * y_dim; i++) {
      if (fscanf(fp, "%d", &value) != 1) {
        fprintf(stderr, "Error: No se pudieron leer todos los pixeles (leídos: %d de %d)\n", 
                i, x_dim * y_dim);
        delete[] pixels;
        fclose(fp);
        exit(1);
      }
      pixels[i] = (unsigned char)value;
    }
  }

  fclose(fp);
}

