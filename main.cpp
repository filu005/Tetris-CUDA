#include <iostream>
#include <ctime>
#include <thread>

//#include <vld.h>

// GLEW
//#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// GLM Mathemtics
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// GL includes
#include "Shader.hpp"

// My includes
#include "Game.hpp"

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void render(GLFWwindow* window);
double calcFPS(GLFWwindow* window, double theTimeInterval = 1.0, std::string theWindowTitle = "NONE");
bool secElapse(double interval = 1.0);

Game* game;

// The MAIN function, from here we start our application and run our Game loop
int main(int argc, char* argv[])
{
	//VLDEnable();
	int widthInBlocks = 10, heightInBlocks = 20;
	GLfloat blockSize = 20.f;
	if(argc >= 3)
	{
		widthInBlocks = atoi(argv[1]);
		heightInBlocks = atoi(argv[2]);
	}

	srand((unsigned int) (time(0)));

	// Init GLFW
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow((int)(widthInBlocks * blockSize), (int)(heightInBlocks * blockSize), "LearnOpenGL", nullptr, nullptr); // Windowed
	glfwMakeContextCurrent(window);

	// Set the required callback functions
	glfwSetKeyCallback(window, key_callback);

	// Initialize GLEW to setup the OpenGL Function pointers
	glewExperimental = GL_TRUE;
	glewInit();

	game = new Game(widthInBlocks, heightInBlocks, blockSize);
	if(argc >= 4)
		game->setLookAheadTetrominosNumbler(atoi(argv[3]));
	if(argc >= 5)
		game->setTetrominoDrawStyle(atoi(argv[4]));
	game->turnOnLogging();

	int numOfIterations = 0;
	if(argc >= 6)
		numOfIterations = atoi(argv[5]);

	// Define the viewport dimensions
	glViewport(0, 0, (GLsizei) (widthInBlocks * blockSize), (GLsizei) (heightInBlocks * blockSize));

	static double start_time = glfwGetTime();
	//int mul = 200;

	// Game loop
	while(!glfwWindowShouldClose(window))
	{
		// Check and call events
		glfwPollEvents();

		calcFPS(window, 1.0, "");

		if(secElapse(game->interval))
		{
			game->tick();

			render(window);
		}

		//http://stackoverflow.com/questions/17138579/idle-thread-time-in-glfw-3-0-c
		std::this_thread::yield();

		if((numOfIterations != 0) && (game->iterations >= numOfIterations))
			glfwSetWindowShouldClose(window, GL_TRUE);

		//int iters = game->getPoints();
		//if(iters != 0 && (iters % mul) == 0)
		//{
		//	std::cout << iters << "\t" << glfwGetTime() - start_time << std::endl;
		//	mul = mul * 2;
		//}
	}

	std::cout << game->getPoints() << "\t" << glfwGetTime() - start_time << std::endl;
	delete game;
	glfwTerminate();
	//VLDReportLeaks();
	return 0;
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	if(action == GLFW_PRESS || action == GLFW_REPEAT)
	{
		switch(key)
		{
		case GLFW_KEY_LEFT:
			game->keyEvent(Game::LEFT);
			break;
		case GLFW_KEY_RIGHT:
			game->keyEvent(Game::RIGHT);
			break;
		case GLFW_KEY_UP:
			game->keyEvent(Game::UP);
			break;
		case GLFW_KEY_DOWN:
			game->keyEvent(Game::DOWN);
			break;
		case GLFW_KEY_ENTER:
			game->switchAIState();
			break;
		case GLFW_KEY_MINUS:
			game->interval *= 2.0;
			break;
		case GLFW_KEY_EQUAL:
			game->interval *= 0.5;
			break;
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		}
	}
	render(window);
}

void render(GLFWwindow* window)
{
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	game->paint();

	// Swap the buffers
	glfwSwapBuffers(window);
}

bool secElapse(double interval)
{
	static double t0 = glfwGetTime();

	double currentTime = glfwGetTime();

	if((currentTime - t0) > interval)
	{
		t0 = glfwGetTime();
		return true;
	}
	else
		return false;
}

// http://r3dux.org/2012/07/a-simple-glfw-fps-counter/
double calcFPS(GLFWwindow* window, double theTimeInterval, std::string theWindowTitle)
{
	// Static values which only get initialised the first time the function runs
	static double t0Value = glfwGetTime(); // Set the initial time to now
	static int    fpsFrameCount = 0;             // Set the initial FPS frame count to 0
	static double fps = 0.0;           // Set the initial FPS value to 0.0

	// Get the current time in seconds since the program started (non-static, so executed every time)
	double currentTime = glfwGetTime();

	// Ensure the time interval between FPS checks is sane (low cap = 0.1s, high-cap = 10.0s)
	// Negative numbers are invalid, 10 fps checks per second at most, 1 every 10 secs at least.
	if(theTimeInterval < 0.1)
	{
		theTimeInterval = 0.1;
	}
	if(theTimeInterval > 10.0)
	{
		theTimeInterval = 10.0;
	}

	// Calculate and display the FPS every specified time interval
	if((currentTime - t0Value) > theTimeInterval)
	{
		// Calculate the FPS as the number of frames divided by the interval in seconds
		fps = (double) fpsFrameCount / (currentTime - t0Value);

		// If the user specified a window title to append the FPS value to...
		if(theWindowTitle != "NONE")
		{
			// Convert the fps value into a string using an output stringstream
			std::ostringstream stream;
			stream << fps;
			std::string fpsString = stream.str();

			// Append the FPS value to the window title details
			theWindowTitle += "FPS: " + fpsString;

			// Convert the new window title to a c_str and set it
			const char* pszConstString = theWindowTitle.c_str();
			glfwSetWindowTitle(window, pszConstString);
		}
		else // If the user didn't specify a window to append the FPS to then output the FPS to the console
		{
			std::cout << "FPS: " << fps << std::endl;
		}

		// Reset the FPS frame counter and set the initial time to be now
		fpsFrameCount = 0;
		t0Value = glfwGetTime();
	}
	else // FPS calculation time interval hasn't elapsed yet? Simply increment the FPS frame counter
	{
		fpsFrameCount++;
	}

	// Return the current FPS - doesn't have to be used if you don't want it!
	return fps;
}