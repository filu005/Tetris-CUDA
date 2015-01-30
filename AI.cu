#include <iomanip>
#include "AI.cuh"

#define CUDA_SAFE(x) do { if (( x ) != cudaSuccess) { \
						 std::cout << "Error at " << __FILE__ << " : " \
							  << __LINE__ << std::endl \
							  << cudaGetErrorString( x ) << std::endl;\
					}} while(0)

__host__
AI::AI(void) : _isRunning(AI::STOP)
{

}

__device__ __constant__
Board d_board;

__host__
std::pair<Tetromino, double> AI::itsShowtime(Board& board, std::vector<Tetromino> tetrominos, unsigned int currentTetrominoIdx)
{
	// host data
	// InData h_startAndEndXPositionPerRotation[4] = { { -1, -1 } };
	InData* h_startAndEndXPositionPerRotation;
	h_startAndEndXPositionPerRotation = (InData*)malloc(sizeof(InData) * 4);
	// Result h_resultsTable[4][10];
	Result* h_resultsTable;
	h_resultsTable = (Result*)malloc(sizeof(Result) * 4 * 10);

	// device data
	InData* d_startAndEndXPositionPerRotation;
	Result* d_resultsTable;

	// init host data
	for(int rotation = 0; rotation < 4; ++rotation)
	{
		h_startAndEndXPositionPerRotation[rotation] = { -1, -1 };
		for(int i = 0; i < 10; ++i)
			h_resultsTable[rotation * 10 + i] = { Tetromino(), -1000000.0 };
	}

	for(int rotation = 0; rotation < 4; ++rotation)
	{
		Tetromino t = tetrominos[currentTetrominoIdx];
		t.rotate(rotation);

		while(board.canMoveLeft(t))
			t.move(-1, 0);

		h_startAndEndXPositionPerRotation[rotation].startX = t.x();

		while(board.canMoveRight(t))
			t.move(1, 0);

		h_startAndEndXPositionPerRotation[rotation].endX = t.x() - 1;
	}

	//for(int i = 0; i < 4; ++i)
	//	std::cout << "rotation: " << i << "; h_startAndEndXPositionPerRotation startX, endX: " << h_startAndEndXPositionPerRotation[i].startX << ", " << h_startAndEndXPositionPerRotation[i].endX << std::endl;
	
	//---------------------------------- board alloc
	// http://stackoverflow.com/questions/17005032/pass-host-pointer-array-to-device-global-memory-pointer-array
	int heightInBlocks = board.getHeight(), widthInBlocks = board.getWidth();
	//const size_t buf_sz = sizeof(bool) * heightInBlocks * widthInBlocks;
	//bool** bboard;
	//bboard = new bool*[heightInBlocks];
	//CUDA_SAFE( cudaMalloc((void **) &bboard[0], buf_sz) );
	//CUDA_SAFE( cudaMemset(bboard[0], (int)false, buf_sz) );
	//for(int i = 1; i < heightInBlocks; i++)
	//	bboard[i] = bboard[i - 1] + widthInBlocks;

	//bool** bboard_;
	//const size_t seq_sz = sizeof(bool*) * size_t(heightInBlocks);
	//CUDA_SAFE( cudaMalloc((void **) &bboard_, seq_sz) );
	//CUDA_SAFE( cudaMemcpy(bboard_, bboard, seq_sz, cudaMemcpyHostToDevice) );

	//for(int i = 0; i < heightInBlocks; ++i)
	//	for(int j = 0; j < widthInBlocks; ++j)
	//		CUDA_SAFE(cudaMemset(&bboard[i][j], (int) board.board[i][j], sizeof(bool)));

	//Board d_board_;
	//d_board_.board = bboard_;
	//d_board_.setWH(widthInBlocks, heightInBlocks);
	//---------------------------------- ~board alloc


	//----
	const size_t buf_sz = sizeof(bool) * heightInBlocks * widthInBlocks;
	const size_t seq_sz = sizeof(bool*) * size_t(heightInBlocks);
	// Gpu memory for sequences - cala pamiec 22x10xbool
	bool *_buf;
	CUDA_SAFE(cudaMalloc((void **) &_buf, buf_sz));

	// Host array for holding sequence device pointers
	bool **seq = new bool*[heightInBlocks];
	size_t offset = 0;
	for(int i = 0; i<heightInBlocks; i++, offset += widthInBlocks)
	{
		seq[i] = _buf + offset;
	}

	// Device array holding sequence pointers
	bool **_seq;
	CUDA_SAFE(cudaMalloc((void **) &_seq, seq_sz));
	CUDA_SAFE(cudaMemcpy(_seq, seq, seq_sz, cudaMemcpyHostToDevice));

	CUDA_SAFE(cudaMemcpy(_buf, board.board[0], buf_sz, cudaMemcpyHostToDevice));

	Board d_board_;
	d_board_.board = _seq;
	d_board_.setWH(widthInBlocks, heightInBlocks);
	//----
	

	// allocate device memory
	// cudaMallocPitch(&doublePointerToData, &pitch, columns*sizeof(DataType), rows)
	size_t pitch_result;// pitch_indata;
	CUDA_SAFE( cudaMalloc(&d_startAndEndXPositionPerRotation, 4 * sizeof(InData)) );
	// CUDA_SAFE(cudaMallocPitch(&d_startAndEndXPositionPerRotation, &pitch_indata, sizeof(InData), 4));
	CUDA_SAFE( cudaMallocPitch(&d_resultsTable, &pitch_result, 10 * sizeof(Result), 4) );

	// przeslij na device:
	// Board
	// startAndEndXPositionPerRotation
	// resultsTable
	// i todo:
	// tetrominos[currentTetrominoIdx]

	// (copy host arrays to device)
	CUDA_SAFE( cudaMemcpyToSymbol(d_board, &d_board_, sizeof(Board)) );
	CUDA_SAFE( cudaMemcpy(d_startAndEndXPositionPerRotation, h_startAndEndXPositionPerRotation, 4 * sizeof(InData), cudaMemcpyHostToDevice) );
	CUDA_SAFE( cudaMemcpy2D(d_resultsTable, pitch_result, h_resultsTable, 10 * sizeof(Result), 10 * sizeof(Result), 4, cudaMemcpyHostToDevice) );


	// wywolaj kernela
	kernel <<< 4, 10 >>>(d_startAndEndXPositionPerRotation, pitch_result, d_resultsTable, tetrominos[currentTetrominoIdx].getName());
	cudaDeviceSynchronize();


	// pobierz tablice z wynikami
	// (copy back device to host matrix)
	//CUDA_SAFE( cudaMemcpy(h_startAndEndXPositionPerRotation, d_startAndEndXPositionPerRotation, 4 * sizeof(InData), cudaMemcpyDeviceToHost) );
	CUDA_SAFE( cudaMemcpy2D(h_resultsTable, 10 * sizeof(Result), d_resultsTable, pitch_result, 10 * sizeof(Result), 4, cudaMemcpyDeviceToHost) );

	// usun wszystko z pamieci device
	//cudaFree(bboard[0]);
	//cudaFree(bboard_);
	cudaFree(_buf);
	cudaFree(_seq);
	cudaFree(d_startAndEndXPositionPerRotation);
	cudaFree(d_resultsTable);

	// przeszukaj resultsTable pod wzgledem najwiekszego resultsTable[i][j].score
	// i zwroc std::make_pair(resultsTable[i][j].t, resultsTable[i][j].score)
	Tetromino bestTetromino;
	double bestScore = -1000000.0;

	for(int rotation = 0; rotation < 4; ++rotation)
	{
		for(int i = 0, idx; i < 10; ++i)
		{
			idx = rotation * 10 + i;
			if(h_resultsTable[idx].score > bestScore)
			{
				bestScore = h_resultsTable[idx].score;
				bestTetromino = h_resultsTable[idx].t;
			}
		}
	}

	// zwolnij pamiec
	//delete[] bboard;
	delete[] seq;
	free(h_resultsTable);
	free(h_startAndEndXPositionPerRotation);

	return std::make_pair(bestTetromino, bestScore);
}

///////////////////////////////////////////////////////////////////////////////////////////// device

__global__
void kernel(InData* startAndEndXPositionPerRotation, size_t pitch_result, Result* resultsTable, int tetrominoName)
{
	const double heightWeight = -0.66569;
	const double linesWeight = 0.99275;
	const double holesWeight = -0.46544;
	const double bumpinessWeight = -0.24077;

	Tetromino t(static_cast<Tetromino::Name>(tetrominoName));
	Board b;
	//-------- board set
	int heightInBlocks = d_board.getHeight(), widthInBlocks = d_board.getWidth();
	bool** pp_board;
	pp_board = new bool*[heightInBlocks];
	pp_board[0] = new bool[heightInBlocks*widthInBlocks];
	for(int i = 1; i < heightInBlocks; i++)
		pp_board[i] = pp_board[i - 1] + widthInBlocks;

	for(int i = 0; i < heightInBlocks; ++i)
		for(int j = 0; j < widthInBlocks; ++j)
			pp_board[i][j] = d_board.board[i][j];

	b.board = pp_board;
	b.setWH(widthInBlocks, heightInBlocks);
	//-------- board set

	int rotation = blockIdx.x;
	int startPosX = startAndEndXPositionPerRotation[rotation].startX;
	int posX = threadIdx.x + startPosX;
	//printf("blockIdx.x: %d\tthreadIdx.x: %d\tposX: %d\n", rotation, threadIdx.x, posX);
	if(posX > startAndEndXPositionPerRotation[rotation].endX)
		return;
	
	t.move(-t.x(), -t.y());//ustawiam tetromino na (0, 0)
	t.move(posX, 0);

	t.rotate(rotation);

	while(b.canMoveDown(t))
		t.move(0, 1);

	b.merge(t);

	double score = heightWeight * b.aggregateHeight() + linesWeight * b.completeLines() + holesWeight * b.holes() + bumpinessWeight * b.bumpiness();

	Result result = { t, score };
	Result* pElement = (Result*) ((char*) resultsTable + blockIdx.x * pitch_result) + threadIdx.x;
	*pElement = result;

	// odkad nie moge miec destruktora ~Board()
	delete[] pp_board[0];
	delete[] pp_board;
}
