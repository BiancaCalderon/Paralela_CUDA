all: pgm.o	hough

hough:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough -std=c++11

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o -std=c++11

clean:
	rm -f pgm.o hough
