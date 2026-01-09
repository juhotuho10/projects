
#ifdef _WIN32
#include "SDL.h"
#elif defined(__APPLE__)
#include "SDL.h"
#else
#include "SDL2/SDL.h"
#endif

#include <stdio.h> // printf
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <CL/cl.h>

int mousePosX;
int mousePosY;

// These are used to decide the window size
#define WINDOW_HEIGHT 512
#define WINDOW_WIDTH  960

#define SIZE WINDOW_WIDTH*WINDOW_HEIGHT

// The number of satellites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites
#define SATELLITE_COUNT 64

// These are used to control the satellite movement
#define SATELLITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000
#define BLACK_HOLE_RADIUS 4.5f

const double dt_factor = (double)DELTATIME / (double)PHYSICSUPDATESPERFRAME;
const double gravity_dt = (double)GRAVITY * (double)DELTATIME / (double)PHYSICSUPDATESPERFRAME;


// Stores 2D data like the coordinates
typedef struct {
	float x;
	float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct {
	double x;
	double y;
} doublevector;

// Each float may vary from 0.0f ... 1.0f
typedef struct {
	float blue;
	float green;
	float red;
} color_f32;

// Stores rendered colors. Each value may vary from 0 ... 255
typedef struct {
	uint8_t blue;
	uint8_t green;
	uint8_t red;
	uint8_t reserved;
} color_u8;




// Pixel buffer which is rendered to the screen
color_u8* pixels;


// Stores the satellite data, which fly around black hole in the space
color_f32 sat_identifiers[SATELLITE_COUNT];
floatvector sat_positions[SATELLITE_COUNT];
floatvector sat_velocitys[SATELLITE_COUNT];



char* readSource(char* kernelPath) {

	cl_int status;
	FILE* fp;
	char* source;
	long int size;

	printf("Program file is: %s\n", kernelPath);

	fp = fopen(kernelPath, "rb");
	if (!fp) {
		printf("Could not open kernel file\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if (status != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}
	size = ftell(fp);
	if (size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}

	rewind(fp);

	source = (char*)malloc(size + 1);

	int i;
	for (i = 0; i < size + 1; i++) {
		source[i] = '\0';
	}

	if (source == NULL) {
		printf("Error allocating space for the kernel source\n");
		exit(-1);
	}

	fread(source, 1, size, fp);
	source[size] = '\0';

	return source;
}


// ## You may add your own variables here ##

cl_int status;

cl_uint numPlatforms = 0;

cl_uint numDevices = 0;
cl_device_id* devices;
cl_context context;
cl_command_queue cmdQueue;
cl_platform_id* platforms = NULL;
cl_mem sat_positions_buffer;
cl_mem sat_identifiers_buffer;
cl_mem pixel_buffer;
cl_kernel kernel;

// global index
size_t globalWorkSize[2];
size_t localWorkSize[2];

cl_program program;

int window_width = WINDOW_WIDTH;
int window_height = WINDOW_HEIGHT;

size_t sat_positions_datasize;
size_t sat_identifiers_datasize;
size_t pixels_datasize;


double tmpPos_x[SATELLITE_COUNT];
double tmpPos_y[SATELLITE_COUNT];

double tmpVel_x[SATELLITE_COUNT];
double tmpVel_y[SATELLITE_COUNT];

unsigned int frameNumber;

void fixedDestroy(void);

cl_int checkError(cl_int status, const char* msg)
{
	if (status != CL_SUCCESS) {
		fprintf(stderr, "error: %s (code %d)\n", msg, status);
		fixedDestroy();
	}
	return status;
}


void init() {

	// ========================= init devices, context, command queue ==========================

	status = clGetPlatformIDs(NULL, NULL, &numPlatforms);
	if (checkError(status, "couldnt get platform ids") != CL_SUCCESS) {
		return status;
	}


	platforms = (cl_platform_id*)malloc(
		numPlatforms * sizeof(cl_platform_id));

	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (checkError(status, "couldnt get platforms") != CL_SUCCESS) {
		return status;
	}

	int device_type = CL_DEVICE_TYPE_GPU;


	status = clGetDeviceIDs(platforms[0], device_type, 0,
		NULL, &numDevices);
	if (checkError(status, "couldnt get device ids") != CL_SUCCESS) {
		return status;
	}

	// allocate enough space for each device
	devices = (cl_device_id*)malloc(
		numDevices * sizeof(cl_device_id));

	// fill in the devices 
	status = clGetDeviceIDs(platforms[0], device_type,
		numDevices, devices, NULL);
	if (checkError(status, "couldnt get platforms") != CL_SUCCESS) {
		return status;
	}

	// create a context and associate it with the devices
	context = clCreateContext(NULL, numDevices, devices, NULL,
		NULL, &status);
	if (checkError(status, "couldnt make context") != CL_SUCCESS) {
		return status;
	}

	cmdQueue = clCreateCommandQueue(context, devices[0], 0,
		&status);
	if (checkError(status, "couldnt create command queue") != CL_SUCCESS) {
		return status;
	}

	// ========================= make program and kernel ==========================


	const char* programSource = readSource("parallel.cl");
	// create a CL program from source
	program = clCreateProgramWithSource(context, 1,
		&programSource, NULL, &status);
	if (checkError(status, "couldnt create program from source") != CL_SUCCESS) {
		return status;
	}

	// compile for the device
	char build_options[256];
	snprintf(build_options, sizeof(build_options),
		"-DNUM_SATELLITES=%d -DWINDOW_WIDTH=%d -DWINDOW_HEIGHT=%d -DSATELLITE_RADIUS=%ff -DBLACK_HOLE_RADIUS=%ff",
		SATELLITE_COUNT, WINDOW_WIDTH, WINDOW_HEIGHT, SATELLITE_RADIUS, BLACK_HOLE_RADIUS);
	status = clBuildProgram(program, numDevices, devices,
		build_options, NULL, NULL);
	if (checkError(status, "couldnt build program") != CL_SUCCESS) {
		return status;
	}

	// create kernels
	kernel = clCreateKernel(program, "parallel", &status);
	if (checkError(status, "couldnt create kernel") != CL_SUCCESS) {
		return status;
	}


	// ========================= init buffers ==========================

	// Compute the size of the data 
	sat_positions_datasize = sizeof(floatvector) * SATELLITE_COUNT;
	sat_identifiers_datasize = sizeof(color_f32) * SATELLITE_COUNT;
	pixels_datasize = sizeof(color_u8) * WINDOW_WIDTH * WINDOW_HEIGHT;

	// Create a buffer object that will contain the data 
	sat_positions_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sat_positions_datasize,
		NULL, &status);

	if (checkError(status, "couldnt create buffer for satellites") != CL_SUCCESS) {
		return status;
	}

	sat_identifiers_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sat_identifiers_datasize,
		NULL, &status);

	if (checkError(status, "couldnt create buffer for satellites") != CL_SUCCESS) {
		return status;
	}

	pixel_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pixels_datasize,
		NULL, &status);
	if (checkError(status, "couldnt create buffer for pixels") != CL_SUCCESS) {
		return status;
	}


	localWorkSize[0] = 8;
	localWorkSize[1] = 8;

	globalWorkSize[0] = ((size_t)WINDOW_WIDTH + localWorkSize[0] - 1) / localWorkSize[0] * localWorkSize[0];
	globalWorkSize[1] = ((size_t)WINDOW_HEIGHT + localWorkSize[1] - 1) / localWorkSize[1] * localWorkSize[1];
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &sat_positions_buffer);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &sat_identifiers_buffer);
	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &pixel_buffer);
	status |= clSetKernelArg(kernel, 5, sizeof(int), &window_width);
	status |= clSetKernelArg(kernel, 6, sizeof(int), &window_height);

	if (checkError(status, "couldnt set kernel arguments") != CL_SUCCESS) {
		return status;
	}
}


// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine)
// Moves the satellites based on gravity
// This is done multiple times in a frame because the Euler integration
// is not accurate enough to be done only once
void parallelPhysicsEngine() {

	int tmpMousePosX = mousePosX;
	int tmpMousePosY = mousePosY;

	// double precision required for accumulation inside this routine,
	// but float storage is ok outside these loops.


	for (int idx = 0; idx < SATELLITE_COUNT; ++idx) {
		tmpPos_x[idx] = sat_positions[idx].x;
		tmpPos_y[idx] = sat_positions[idx].y;
		tmpVel_x[idx] = sat_velocitys[idx].x;
		tmpVel_y[idx] = sat_velocitys[idx].y;
	}
	// Physics iteration loop
	// Cannot be paralellized, the time steps all have to come one after another, they cannot happen at arbitrary steps
	for (int physicsUpdateIndex = 0;
		physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
		++physicsUpdateIndex) {

		// Physics satellite loop
		#pragma loop(hint_parallel(4))
		#pragma loop(ivdep)
		for (int i = 0; i < SATELLITE_COUNT; ++i) {

			double dx = tmpPos_x[i] - tmpMousePosX;
			double dy = tmpPos_y[i] - tmpMousePosY;

			double dist_sq = dx * dx + dy * dy;
			double inv_dist = 1.0 / sqrt(dist_sq);

			double force_factor = inv_dist * inv_dist * inv_dist * gravity_dt;

			// Update velocity 
			tmpVel_x[i] -= dx * force_factor;
			tmpVel_y[i] -= dy * force_factor;

			// Update position based on velocity
			tmpPos_x[i] += tmpVel_x[i] * dt_factor;
			tmpPos_y[i] += tmpVel_y[i] * dt_factor;
		}
	}

	// double precision required for accumulation inside this routine,
	// but float storage is ok outside these loops.
	// copy back the float storage.
	for (int idx2 = 0; idx2 < SATELLITE_COUNT; ++idx2) {
		sat_positions[idx2].x = tmpPos_x[idx2];
		sat_positions[idx2].y = tmpPos_y[idx2];
		sat_velocitys[idx2].x = tmpVel_x[idx2];
		sat_velocitys[idx2].y = tmpVel_y[idx2];
	}

}


void parallelGraphicsEngine() {

	// ========================= run program ================================


	// Write input array A to the device buffer bufferA

	status = clEnqueueWriteBuffer(cmdQueue, sat_positions_buffer, CL_TRUE,
		0, sat_positions_datasize, sat_positions, 0, NULL, NULL);

	if (checkError(status, "couldnt write positions into the buffer") != CL_SUCCESS) {
		return status;
	}

	status = clEnqueueWriteBuffer(cmdQueue, sat_identifiers_buffer, CL_TRUE,
		0, sat_identifiers_datasize, sat_identifiers, 0, NULL, NULL);

	if (checkError(status, "couldnt write indentifiers into the buffer") != CL_SUCCESS) {
		return status;
	}

	status = clSetKernelArg(kernel, 0, sizeof(int), &mousePosX);
	status |= clSetKernelArg(kernel, 1, sizeof(int), &mousePosY);


	if (checkError(status, "couldnt set kernel arguments") != CL_SUCCESS) {
		return status;
	}


	// Execute the kernel for execution
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL,
		globalWorkSize, localWorkSize, 0, NULL, NULL);

	if (checkError(status, "couldnt enqueue kernel") != CL_SUCCESS) {
		return status;
	}

	// ========================= get results from program ================================

	// Read the device output buffer to the host output array
	clEnqueueReadBuffer(cmdQueue, pixel_buffer, CL_FALSE, 0,
		pixels_datasize, pixels, 0, NULL, NULL);

	clFinish(cmdQueue);

}



void destroy() {
	// free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(sat_positions_buffer);
	clReleaseMemObject(sat_identifiers_buffer);
	clReleaseMemObject(pixel_buffer);
	clReleaseContext(context);

	// free host resources
	free(platforms);
	free(devices);

}



////////////////////////////////////////////////
// ¤¤ TO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

/* Parallelization Excercise 2024
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
					  Heikki Kultala heikki.kultala@tut.fi
					  Topi Leppanen  topi.leppanen@tuni.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satellites affect the color with weighted average.
			   add physic correctness check.
VERSION 20.0 - relax physic correctness check
VERSION 24.0 - port to SDL2
VERSION 25.0 - add macOS support
*/



#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)
SDL_Window* win;
SDL_Surface* surf;
// Is used to find out frame times
int totalTimeAcc, satelliteMovementAcc, pixelColoringAcc, frameCount;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;


// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void compute(void) {
	int timeSinceStart = SDL_GetTicks();

	SDL_GetMouseState(&mousePosX, &mousePosY);
	if ((mousePosX == 0) && (mousePosY == 0)) {
		mousePosX = HORIZONTAL_CENTER;
		mousePosY = VERTICAL_CENTER;
	}
	
	parallelPhysicsEngine();

	int satelliteMovementMoment = SDL_GetTicks();
	int satelliteMovementTime = satelliteMovementMoment - timeSinceStart;

	// Decides the colors for the pixels
	parallelGraphicsEngine();

	int pixelColoringMoment = SDL_GetTicks();
	int pixelColoringTime = pixelColoringMoment - satelliteMovementMoment;

	int finishTime = SDL_GetTicks();

	if (frameNumber == 0) {
		previousFinishTime = finishTime;
		printf("Time spent on moving satellites + Time spent on space coloring : Total time in milliseconds between frames (might not equal the sum of the left-hand expression)\n");
	}
	else if (frameNumber > 2) {
		// Print timings
		int totalTime = finishTime - previousFinishTime;
		previousFinishTime = finishTime;

		printf("Latency of this frame %i + %i : %ims \n",
			satelliteMovementTime, pixelColoringTime, totalTime);

		frameCount++;
		totalTimeAcc += totalTime;
		satelliteMovementAcc += satelliteMovementTime;
		pixelColoringAcc += pixelColoringTime;
		printf("Averaged over all frames: %i + %i : %ims.\n",
			satelliteMovementAcc / frameCount, pixelColoringAcc / frameCount, totalTimeAcc / frameCount);

	}
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Probably not the best random number generator
float randomNumber(float min, float max) {
	return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed) {

	if (seed != 0) {
		srand(seed);
	}

	// Init pixel buffer which is rendered to the widow
	pixels = (color_u8*)malloc(sizeof(color_u8) * SIZE);

	// Create random satellites
	for (int i = 0; i < SATELLITE_COUNT; ++i) {

		// Random reddish color
		color_f32 id = { .red = randomNumber(0.f, 0.15f) + 0.1f,
					.green = randomNumber(0.f, 0.14f) + 0.0f,
					.blue = randomNumber(0.f, 0.16f) + 0.0f };

		// Random position with margins to borders
		floatvector initialPosition = { .x = HORIZONTAL_CENTER - randomNumber(50, 320),
								.y = VERTICAL_CENTER - randomNumber(50, 320) };
		initialPosition.x = (i / 2 % 2 == 0) ?
			initialPosition.x : WINDOW_WIDTH - initialPosition.x;
		initialPosition.y = (i < SATELLITE_COUNT / 2) ?
			initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

		// Randomize velocity tangential to the balck hole
		floatvector positionToBlackHole = { .x = initialPosition.x - HORIZONTAL_CENTER,
									  .y = initialPosition.y - VERTICAL_CENTER };
		float distance = (0.06 + randomNumber(-0.01f, 0.01f)) /
			sqrt(positionToBlackHole.x * positionToBlackHole.x +
				positionToBlackHole.y * positionToBlackHole.y);
		floatvector initialVelocity = { .x = distance * -positionToBlackHole.y,
								  .y = distance * positionToBlackHole.x };

		// Every other orbits clockwise
		if (i % 2 == 0) {
			initialVelocity.x = -initialVelocity.x;
			initialVelocity.y = -initialVelocity.y;
		}

		sat_identifiers[i] = id;
		sat_positions[i] = initialPosition;
		sat_velocitys[i] = initialVelocity;

	}
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void fixedDestroy(void) {
	destroy();

	free(pixels);

	if (seed != 0) {
		printf("Used seed: %i\n", seed);
	}
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Renders pixels-buffer to the window
void render(void) {
	SDL_LockSurface(surf);
	memcpy(surf->pixels, pixels, WINDOW_WIDTH * WINDOW_HEIGHT * 4);
	SDL_UnlockSurface(surf);

	SDL_UpdateWindowSurface(win);
	frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits render window and starts mainloop
int main(int argc, char** argv) {

	if (argc > 1) {
		seed = atoi(argv[1]);
		printf("Using seed: %i\n", seed);
	}

	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);
	win = SDL_CreateWindow(
		"Satellites",
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		WINDOW_WIDTH, WINDOW_HEIGHT,
		0
	);
	surf = SDL_GetWindowSurface(win);

	fixedInit(seed);
	init();

	SDL_Event event;
	int running = 1;
	while (running) {
		while (SDL_PollEvent(&event)) switch (event.type) {
		case SDL_QUIT:
			printf("Quit called\n");
			running = 0;
			break;
		}
		compute();
		render();
	}
	SDL_Quit();
	fixedDestroy();
}
