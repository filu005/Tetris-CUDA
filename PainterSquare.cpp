#include "PainterSquare.hpp"

GLfloat PainterSquare::vertices[12] =
{
	// First triangle
	20.0f, 20.0f,	// Top Right
	20.0f, 0.0f,	// Bottom Right
	0.0f, 20.0f, // Top Left 
	// Second triangle
	20.0f, 0.0f,	// Bottom Right
	0.0f, 0.0f, // Bottom Left
	0.0f, 20.0f // Top Left 
};
