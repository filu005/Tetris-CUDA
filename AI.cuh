//AI.cuh
#pragma once
#include <utility>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
//#include <cuda.h>

#include "Tetromino.cuh"
#include "Board.cuh"

struct Result
{
	Tetromino t;
	double score;
};

struct InData
{
	int startX;
	int endX;
};

class AI
{
public:
	enum State { STOP = 0, START = 1 };
	__host__ AI();

	__host__ std::pair<Tetromino, double> itsShowtime(Board& board, std::vector<Tetromino> tetrominos, unsigned int currentTetrominoIdx);
	__host__ void flipState() { _isRunning = (_isRunning == State::STOP) ? State::START : State::STOP; }
	__host__ bool isRunning() const { return (_isRunning == State::START); }

private:
	const double heightWeight = -0.66569;
	const double linesWeight = 0.99275;
	const double holesWeight = -0.46544;
	const double bumpinessWeight = -0.24077;

	State _isRunning;
};

__global__ void kernel(InData* startAndEndXPositionPerRotation, size_t pitch_result, Result* resultsTable, int tetrominoName);