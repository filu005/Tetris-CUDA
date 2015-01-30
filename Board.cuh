//Board.cuh
#pragma once
#include <cuda_runtime.h>

#include "PainterSquare.hpp"

class Painter;

class Board
{
public:
	__host__ __device__ Board() { };
	__host__ __device__ Board(int widthInBlocks, int heightInBlocks);
	__host__ __device__ Board(const Board& b);
	//__host__ __device__ ~Board();

	__host__ __device__ void paint(Painter& p) const;
	__host__ __device__ bool collide(const Tetromino& t) const;
	__host__ __device__ bool canMoveDown(Tetromino t) const;
	__host__ __device__ bool canMoveLeft(Tetromino t) const;
	__host__ __device__ bool canMoveRight(Tetromino t) const;
	__host__ __device__ int removeLines();
	__host__ __device__ void merge(const Tetromino& t);

	// Heurystyki
	__host__ __device__ int aggregateHeight() const;
	__host__ __device__ int completeLines() const;
	__host__ __device__ int holes() const;
	__host__ __device__ int bumpiness() const;

	__host__ __device__ int getColHeight(int col) const;

	__host__ __device__ int getWidth() const { return _widthInBlocks; }
	__host__ __device__ int getHeight() const { return _heightInBlocks; }

	__host__ __device__ void setWH(int width, int height)
	{
		_widthInBlocks = width;
		_heightInBlocks = height;
	}

	bool** board;

//private:
	int _widthInBlocks;// = 10;
	int _heightInBlocks;// = 22;
};
