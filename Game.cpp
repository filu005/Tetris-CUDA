#include "Game.hpp"

Game::Game(int widthInBlocks, int heightInBlocks, GLfloat blockSize) : iterations(0), interval(1.0), _points(0), _doublePoints(0), _lookAheadTetrominos(0), _tetrominoDrawStyle(1), _placedTetrominos(0), _logging(false), _widthInBlocks(widthInBlocks), _heightInBlocks(heightInBlocks)
{
	// set the heap size for device size new/delete to 128 MB
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * (1 << 20));

	_board = std::make_shared<Board>(widthInBlocks, heightInBlocks);
	_tetrominos.reserve(_lookAheadTetrominos + 1);
	_tetrominoPool.reserve(7);
	fillTetrominoPool();

	for(int i = 0; i < _lookAheadTetrominos + 1; ++i)
		_tetrominos.push_back(randTetrominoFromPool());

	_painter.setProjectionHeight((GLfloat) (heightInBlocks * blockSize));
	_painter.setProjectionWidth((GLfloat) (widthInBlocks * blockSize));
}

void Game::paint(void)
{
	_board->paint(_painter);
	_tetrominos[0].paint(_painter);
}

void Game::tick(void)
{
	Tetromino curTet = _tetrominos[0];
	if(_ai.isRunning())
	{
		std::vector<Tetromino> aiTetrominos;
		aiTetrominos.reserve(_lookAheadTetrominos + 1);
		for(int i = 0; i < _lookAheadTetrominos + 1; ++i)
			aiTetrominos.push_back(_tetrominos[i]);

		auto result = _ai.itsShowtime(*_board, aiTetrominos, 0);
		curTet = result.first;
	}
	Tetromino t = curTet;
	t.move(0, 1);
	if(_board->collide(t))
	{
		_board->merge(curTet);
		int removeLinesNum = _board->removeLines();
		if(removeLinesNum > 0)
		{
			_points += removeLinesNum;
			if(removeLinesNum > 1)
				_doublePoints++;
			//std::cout << "wynik: " << _points << "\n";
		}

		for(int i = 0; i < _lookAheadTetrominos; ++i)
			_tetrominos[i] = _tetrominos[i+1];

		_tetrominos[_lookAheadTetrominos] = randTetrominoFromPool();

		if(_board->collide(_tetrominos[0]))
			restart();
	}
	else
		_tetrominos[0] = t;
}

Tetromino Game::randTetrominoFromPool(void)
{
	Tetromino rTet(Tetromino::I);

	fillTetrominoPool();

	int tIdx = rand() % _tetrominoPool.size();
	rTet = _tetrominoPool[tIdx];
	_tetrominoPool.erase(_tetrominoPool.begin() + tIdx);
	return rTet;
}

void Game::keyEvent(Direction d)
{
	Tetromino t = _tetrominos[0];
	switch(d)
	{
	case UP: t.rotate(1);
		break;
	case DOWN: t.move(0, 1);
		break;
	case LEFT: t.move(-1, 0);
		break;
	case RIGHT: t.move(1, 0);
		break;
	}
	if(!_board->collide(t))
		_tetrominos[0] = t;
}

void Game::fillTetrominoPool(void)
{
	if(_tetrominoPool.size() <= 0)
	{
		for(int i = 0; i < 7; ++i)
		{
			// last minute branching!!
			if(_tetrominoDrawStyle == 0)
				_tetrominoPool.push_back(Tetromino(static_cast<Tetromino::Name>(rand() % 7)));
			else
				_tetrominoPool.push_back(Tetromino(static_cast<Tetromino::Name>(i)));
		}
	}
}

void Game::restart(void)
{
	std::cout << "end game. wynik: " << _points << "\n";
	++iterations;
	_ofstrm << _points << "\t" << _doublePoints << "\t" << _placedTetrominos << std::endl;
	_points = _doublePoints = _placedTetrominos = 0;
	_board = make_shared<Board>(_board->getWidth(), _board->getHeight() - 2);

	_tetrominos.erase(_tetrominos.begin(), _tetrominos.end());
	_tetrominoPool.erase(_tetrominoPool.begin(), _tetrominoPool.end());
	fillTetrominoPool();
	for(int i = 0; i < _lookAheadTetrominos + 1; ++i)
		_tetrominos.push_back(randTetrominoFromPool());
}

void Game::turnOnLogging(void)
{
	_logging = true;
	std::string filename("log(");
	filename += std::to_string(_widthInBlocks);
	filename += "x";
	filename += std::to_string(_heightInBlocks);
	filename += ")[";
	filename += std::to_string(_lookAheadTetrominos);
	filename += "][";
	filename += std::to_string(_tetrominoDrawStyle);
	filename += "].txt";
	_ofstrm.open(filename.c_str(), std::ios::out);
	if(!_ofstrm.good())
		std::cout << "nie udalo sie utworzyc pliku " << filename << "\n";

	_ofstrm << "[" << _widthInBlocks << ", " << _heightInBlocks << "]" << std::endl;
}

