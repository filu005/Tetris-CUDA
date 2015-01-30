#pragma once
#include <memory>
#include <vector>
#include <fstream>

#include "Tetromino.cuh"
#include "Painter.hpp"
#include "Board.cuh"
#include "AI.cuh"

#include <GL/glew.h>

class Game
{
public:
	enum Direction { UP, DOWN, LEFT, RIGHT };
	Game(int widthInBlocks, int heightInBlocks, GLfloat blockSize);
	~Game()
	{
		_ofstrm << std::flush;
		_ofstrm.close();
	}

	void tick();
	void paint();
	void keyEvent(Direction);
	void fillTetrominoPool();
	Tetromino randTetrominoFromPool();
	void restart();

	void switchAIState() { _ai.flipState(); }
	void turnOnLogging();
	void setLookAheadTetrominosNumbler(int num) { _lookAheadTetrominos = num; }
	void setTetrominoDrawStyle(int num) { _tetrominoDrawStyle = (num <= 1) ? num : 1; }

	int iterations;
	double interval;

private:
	int _points, _doublePoints;
	int _lookAheadTetrominos;
	int _tetrominoDrawStyle;
	int _placedTetrominos;
	Painter _painter;
	std::shared_ptr<Board> _board;
	std::vector<Tetromino> _tetrominos;
	std::vector<Tetromino> _tetrominoPool;
	AI _ai;
	std::ofstream _ofstrm;
	bool _logging;
	int _widthInBlocks, _heightInBlocks;
};
