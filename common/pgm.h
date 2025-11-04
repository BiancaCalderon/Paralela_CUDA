#ifndef PGM_H
#define PGM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class PGMImage {
public:
  int x_dim, y_dim;
  unsigned char *pixels;

  PGMImage(const char *filename);
  ~PGMImage();
  
private:
  void readPGM(const char *filename);
};

#endif // PGM_H

