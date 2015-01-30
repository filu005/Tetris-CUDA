#pragma once
#include <GL/glew.h>

class PainterSquare
{
public:
	PainterSquare() : _squareSize(20.f)
	{
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		// Position attribute
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*) 0);
		glEnableVertexAttribArray(0);

		glBindVertexArray(0);
	}

	~PainterSquare()
	{
		glDisableVertexAttribArray(0);
		glDeleteBuffers(1, &VBO);
		glDeleteVertexArrays(1, &VAO);
	}

	void setSquareSize(GLfloat squareSize)
	{
		for(int i = 0; i < 12; ++i)
		{
			vertices[i] /= _squareSize;
			vertices[i] *= squareSize;
		}
		_squareSize = squareSize;
	}

	GLuint getVBO() const { return VBO; }
	GLuint getVAO() const { return VAO; }

private:
	GLuint VBO, VAO;
	GLfloat _squareSize;
	static GLfloat vertices[12];
};
