#include "Painter.hpp"
#include "Tetromino.cuh"
//
//__device__ const char* shapes[7] =
//{
//	"  8 " // I
//	"  8 "
//	"  8 "
//	"  8 ",
//
//	"  8 " // J
//	"  8 "
//	" 88 "
//	"    ",
//
//	" 8  " // L
//	" 8  "
//	" 88 "
//	"    ",
//
//	"    " // O
//	" 88 "
//	" 88 "
//	"    ",
//
//	"  8 " // S
//	" 88 "
//	" 8  "
//	"    ",
//
//	" 8  " // Z
//	" 88 "
//	"  8 "
//	"    ",
//
//	"    " // T
//	" 888"
//	"  8 "
//	"    "
//};

__host__ __device__ Tetromino::Tetromino() : _name(Name::I), _x(0), _y(0), _angle(0)
{
	
}

__host__ __device__ Tetromino::Tetromino(Name name) : _name(name), _x(5), _y(0), _angle(0)
{

}

__host__ void Tetromino::paint(Painter& p) const
{
	p.paint(*this);
}

__host__ __device__ unsigned int Tetromino::getName(void) const
{
	return _name;
}

__host__ __device__ bool Tetromino::map(int x, int y) const
{
	const struct
	{
		int x, y;
	} ROTATE[][16] = {
		{
			{ 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 },
			{ 1, 0 }, { 1, 1 }, { 1, 2 }, { 1, 3 },
			{ 2, 0 }, { 2, 1 }, { 2, 2 }, { 2, 3 },
			{ 3, 0 }, { 3, 1 }, { 3, 2 }, { 3, 3 }
		},
		{
			{ 3, 0 }, { 2, 0 }, { 1, 0 }, { 0, 0 },
			{ 3, 1 }, { 2, 1 }, { 1, 1 }, { 0, 1 },
			{ 3, 2 }, { 2, 2 }, { 1, 2 }, { 0, 2 },
			{ 3, 3 }, { 2, 3 }, { 1, 3 }, { 0, 3 }
		},
		{
			{ 3, 3 }, { 3, 2 }, { 3, 1 }, { 3, 0 },
			{ 2, 3 }, { 2, 2 }, { 2, 1 }, { 2, 0 },
			{ 1, 3 }, { 1, 2 }, { 1, 1 }, { 1, 0 },
			{ 0, 3 }, { 0, 2 }, { 0, 1 }, { 0, 0 }
		},
		{
			{ 0, 3 }, { 1, 3 }, { 2, 3 }, { 3, 3 },
			{ 0, 2 }, { 1, 2 }, { 2, 2 }, { 3, 2 },
			{ 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 },
			{ 0, 0 }, { 1, 0 }, { 2, 0 }, { 3, 0 }
		}
	};

	const char* shapes[7] =
	{
		"  8 " // I
		"  8 "
		"  8 "
		"  8 ",

		"  8 " // J
		"  8 "
		" 88 "
		"    ",

		" 8  " // L
		" 8  "
		" 88 "
		"    ",

		"    " // O
		" 88 "
		" 88 "
		"    ",

		"  8 " // S
		" 88 "
		" 8  "
		"    ",

		" 8  " // Z
		" 88 "
		"  8 "
		"    ",

		"    " // T
		" 888"
		"  8 "
		"    "
	};
	bool val = shapes[_name]
		[ROTATE[_angle][y * 4 + x].y * 4 + ROTATE[_angle][y * 4 + x].x] != ' ';

	return val;
}

__host__ __device__ void Tetromino::move(int x, int y)
{
	_x += x;
	_y += y;
}

__host__ __device__ void Tetromino::rotate(int angle)
{
	_angle = (_angle + angle + 4) % 4;
}

