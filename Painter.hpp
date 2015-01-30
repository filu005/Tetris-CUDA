#pragma once
#include "Shader.hpp"
#include "PainterSquare.hpp"

class Tetromino;
class Board;

class Painter
{
public:
	enum Color { CYAN = 0, YELLOW, PURPLE, GREEN, RED, BLUE, ORANGE, WHITE, BLACK };

	Painter();
	void setProjectionWidth(float projWidth) { _projectionWidth = projWidth; }
	void setProjectionHeight(float projHeight) { _projectionHeight = projHeight; }
	void paint(const Tetromino& t);
	void paint(const Board& b);

	void setColor(Color color);

private:
	Shader shader;
	PainterSquare squarePainter;

	float _projectionWidth;
	float _projectionHeight;
	static GLfloat colorValues[9][3];
};
