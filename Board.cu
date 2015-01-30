#include "Painter.hpp"
#include "Tetromino.cuh"
#include "Board.cuh"

__host__ __device__
Board::Board(int widthInBlocks, int heightInBlocks) : _widthInBlocks(widthInBlocks), _heightInBlocks(heightInBlocks + 2)
{
	// alokacja ciaglego obszaru pamieci dla tablicy 3d
	// T ***m_buffer = new T**[xMax];
	//	T  **m_tempxy = new T*[xMax*yMax];
	//	T   *m_tempxyz = new T[xMax*yMax*zMax];
	//	for(int x = 0; x < xMax; x++, m_tempxy += yMax) {
	//		m_buffer[x] = m_tempxy;
	//		for(int y = 0; y < yMax; y++, m_tempxyz += zMax) {
	//			m_buffer[x][y] = m_tempxyz;
	//		}
	//	}
	//	i dla 2d:
	board = new bool*[_heightInBlocks];
	board[0] = new bool[_heightInBlocks*_widthInBlocks];
	for(int i = 1; i < _heightInBlocks; i++)
		board[i] = board[i - 1] + _widthInBlocks;

	for(int i = 0; i < _heightInBlocks; ++i)
		for(int j = 0; j < _widthInBlocks; ++j)
			board[i][j] = false;
	//std::fill(&board[0][0], &board[22][0], false); // [1]
}

__host__ __device__
Board::Board(const Board& b) : _widthInBlocks(b._widthInBlocks), _heightInBlocks(b._heightInBlocks)
{
	board = new bool*[_heightInBlocks];
	board[0] = new bool[_heightInBlocks*_widthInBlocks];
	for(int i = 1; i < _heightInBlocks; i++)
		board[i] = board[i - 1] + _widthInBlocks;

	for(int i = 0; i < _heightInBlocks; ++i)
		for(int j = 0; j < _widthInBlocks; ++j)
			this->board[i][j] = b.board[i][j];
}

//__host__ __device__
//Board::~Board(void)
//{
//	// dealokacja pamieci 3d:
//	// delete [] m_buffer[0][0];
//	//	delete[] m_buffer[0];
//	//	delete[] m_buffer;
//	//	i 2d:
//	delete[] board[0];
//	delete[] board;
//}

__host__ __device__
void Board::paint(Painter& p) const
{
	p.paint(*this);
}

__host__ __device__
bool Board::collide(const Tetromino& t) const
{
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			if(t.map(j, i))
			{
				// odejmuje 2 bo polozenie t[x,y] mam ustawione w srodku figury. albo jakos tak
				int mx = i + t.x() - 2, my = j + t.y();
				if(mx < 0 || mx >= _widthInBlocks || my >= _heightInBlocks)
					return true;
				if(board[my][mx] == true)
					return true;
			}
		}
	}
	return false;
}

__host__ __device__
int Board::removeLines(void)
{
	int points = 0;

	for(int i = 0; i < _heightInBlocks; ++i)
	{
		bool solidLine = true;
		for(int j = 0; j < _widthInBlocks; ++j)
		{
			if(board[i][j])
				continue;
			else
				solidLine = false;
		}
		if(solidLine)
		{
#ifdef DEBUG
			std::cout << "zjadam linie\n";
#endif
			for(int k = i; k > 0; --k)
				for(int j = 0; j < _widthInBlocks; ++j)
					board[k][j] = board[k - 1][j];

			points++;
		}
	}

	return points;
}

__host__ __device__
void Board::merge(const Tetromino& t)
{
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			int mx = i + t.x() - 2, my = j + t.y();
			if(mx >= 0 && mx < _widthInBlocks && my >= 0 && my < _heightInBlocks)
				board[my][mx] = board[my][mx] || t.map(j, i);
		}
	}
}

__host__ __device__
int Board::aggregateHeight(void) const
{
	int sum = 0;
	for(int i = 0; i < _widthInBlocks; ++i)
		sum += getColHeight(i);
	return sum;
}

__host__ __device__
int Board::completeLines(void) const
{
	int sum = 0;
	for(int i = 0; i < _heightInBlocks; ++i)
	{
		bool solidLine = true;
		for(int j = 0; j < _widthInBlocks; ++j)
		{
			if(board[i][j])
				continue;
			else
				solidLine = false;
		}
		if(solidLine)
			++sum;
	}

	return sum;
}

__host__ __device__
int Board::holes(void) const
{
	int sum = 0;

	for(int i = 0; i < _widthInBlocks; ++i)
	{
		bool block = false;
		for(int j = 0; j < _heightInBlocks; ++j)
		{
			if(board[j][i])
				block = true;
			else if(board[j][i] == false && block)
				++sum;
		}
	}
	return sum;
}

__host__ __device__
int Board::bumpiness(void) const
{
	int sum = 0;

	for(int i = 0; i < _widthInBlocks - 1; ++i)
		sum += abs(getColHeight(i) - getColHeight(i + 1));

	return sum;
}

__host__ __device__
int Board::getColHeight(int col) const
{
	int res = 0;
	for(; res < _heightInBlocks && (board[res][col] == false); ++res);
	return (_heightInBlocks - res);
}

__host__ __device__
bool Board::canMoveDown(Tetromino t) const
{
	t.move(0, 1);
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			if(t.map(j, i))
			{
				// odejmuje 2 bo polozenie t[x,y] mam ustawione w srodku figury. albo jakos tak
				int mx = i + t.x() - 2, my = j + t.y();
				if(my >= _heightInBlocks || (board[my][mx] == true))
					return false;
			}
		}
	}
	return true;
}

__host__ __device__
bool Board::canMoveRight(Tetromino t) const
{
	t.move(1, 0);
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			if(t.map(j, i))
			{
				// odejmuje 2 bo polozenie t[x,y] mam ustawione w srodku figury. albo jakos tak
				int mx = i + t.x() - 2, my = j + t.y();
				if(mx > _widthInBlocks || (board[my][mx] == true))
					return false;
			}
		}
	}
	return true;
}

__host__ __device__
bool Board::canMoveLeft(Tetromino t) const
{
	t.move(-1, 0);
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			if(t.map(j, i))
			{
				// odejmuje 2 bo polozenie t[x,y] mam ustawione w srodku figury. albo jakos tak
				int mx = i + t.x() - 2, my = j + t.y();
				if(my >= _heightInBlocks)
					return false;
				if(mx < 0 || (board[my][mx] == true))
					return false;
			}
		}
	}
	return true;
}

