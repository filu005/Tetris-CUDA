#include "Tetromino.cuh"
#include "Board.cuh"
#include "Painter.hpp"

// GLM Mathemtics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


GLfloat Painter::colorValues[9][3] =
{
	{ 0.0f, 1.0f, 1.0f },
	{ 1.0f, 1.0f, 0.0f },
	{ 0.5f, 0.0f, 0.5f },
	{ 0.0f, 1.0f, 0.0f },
	{ 1.0f, 0.0f, 0.0f },
	{ 0.0f, 0.0f, 1.0f },
	{ 1.0f, 0.64f, 0.0f },
	{ 1.0f, 1.0f, 1.0f },
	{ 0.0f, 0.0f, 0.0f }
};

Painter::Painter() : shader(Shader("./shaders/default.vs", "./shaders/default.frag")), _projectionWidth(200.f), _projectionHeight(400.f)
{

}

void Painter::paint(const Tetromino& t)
{
	const auto& VAO = squarePainter.getVAO();
	shader.Use();

	// Create transformations
	glm::mat4 view;
	glm::mat4 projection;

	//view = glm::translate(view, glm::vec3(0.0f, 0.0f, -2.0f));
	projection = glm::ortho(0.0f, _projectionWidth, _projectionHeight, 0.0f, -0.1f, 1000.0f);
	// Get their uniform location
	GLint viewLoc = glGetUniformLocation(shader.Program, "view");
	GLint projLoc = glGetUniformLocation(shader.Program, "projection");
	// Pass them to the shaders
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

	setColor(static_cast<Painter::Color>(t.getName()));

	glBindVertexArray(VAO);

	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			if(t.map(j, i))
			{
				GLfloat fx = (t.x() + i - 2)*20.f, fy = (t.y() + j - 2)*20.f;
				glm::mat4 model;
				model = glm::translate(model, glm::vec3(fx, fy, 0.0f));
				model = glm::scale(model, glm::vec3(0.95f, 0.95f, 1.0f));

				GLint modelLoc = glGetUniformLocation(shader.Program, "model");
				glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

				glDrawArrays(GL_TRIANGLES, 0, 6);
			}
		}
	}
	glBindVertexArray(0);
}

void Painter::paint(const Board& b)
{
	const auto& VAO = squarePainter.getVAO();
	shader.Use();

	// Create transformations
	glm::mat4 view;
	glm::mat4 projection;

	//view = glm::translate(view, glm::vec3(0.0f, 0.0f, -2.0f));
	projection = glm::ortho(0.0f, _projectionWidth, _projectionHeight, 0.0f, -0.1f, 1000.0f);
	// Get their uniform location
	GLint viewLoc = glGetUniformLocation(shader.Program, "view");
	GLint projLoc = glGetUniformLocation(shader.Program, "projection");
	// Pass them to the shaders
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

	glBindVertexArray(VAO);

	for(int i = 2; i < b.getHeight(); ++i)
	{
		for(int j = 0; j < b.getWidth(); ++j)
		{
			GLfloat fx = j*20.f, fy = (i - 2)*20.f;
			glm::mat4 model;
			model = glm::translate(model, glm::vec3(fx, fy, 0.0f));
			model = glm::scale(model, glm::vec3(0.95f, 0.95f, 1.0f));

			GLint modelLoc = glGetUniformLocation(shader.Program, "model");
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

			if(b.board[i][j])
				setColor(Painter::WHITE);
			else
				setColor(Painter::BLACK);

			glDrawArrays(GL_TRIANGLES, 0, 6);
		}
	}
	glBindVertexArray(0);
}

void Painter::setColor(Color color)
{
	GLint vertexColorLocation = glGetUniformLocation(shader.Program, "color");
	glUniform3f(vertexColorLocation, colorValues[color][0], colorValues[color][1], colorValues[color][2]);
}

