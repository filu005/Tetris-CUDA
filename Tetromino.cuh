// Tetroino.cuh
#pragma once
#include <cuda_runtime.h>

#include "PainterSquare.hpp"

class Painter;

// http://tetris.wikia.com/wiki/Tetris_Guideline

class Tetromino
{
public:
	enum Name { I, O, T, S, Z, J, L };

	__host__ __device__ Tetromino();
	__host__ __device__ Tetromino(Name name);
	__host__ __device__ ~Tetromino() { }

	__host__ void paint(Painter& p) const;
	__host__ __device__ bool map(int x, int y) const;
	__host__ __device__ void move(int x, int y);
	__host__ __device__ void rotate(int angle);
	__host__ __device__ unsigned int getName() const;
	__host__ __device__ unsigned int x() const { return _x; }
	__host__ __device__ unsigned int y() const { return _y; }

private:
	unsigned int _name;
	unsigned int _x, _y;
	int _angle;

	//const char* shapes[7];
};