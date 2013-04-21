/*
 * utils.cpp
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#include "utils.h"
#include "constants.h"

using namespace std;
using namespace cv;

int mygetch()
{
    struct termios oldt,
    newt;
    int ch;
    tcgetattr( STDIN_FILENO, &oldt );
    newt = oldt;
    newt.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newt );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldt );
    return ch;
}


void GenerateFeatures(std::vector<Feature>& features)
{
    int offset = W_WIDTH;
    for( int x = 0; x < W_WIDTH; x++ )
    {
        for( int y = 0; y < W_HEIGHT; y++ )
        {
            for( int dx = 1; dx < W_WIDTH; dx++ )
            {
                for( int dy = 1; dy < W_HEIGHT; dy++ )
                {
                    // haar_x2
                    if ( (x+dx*2 < W_WIDTH) && (y+dy < W_HEIGHT) )
                    {
                        features.push_back( Feature( offset,
                            x,    y, dx*2, dy, -1,
                            x+dx, y, dx  , dy, +2 ) );
                    }
                    // haar_y2
                    if ( (x+dx < W_WIDTH) && (y+dy*2 < W_HEIGHT) )
                    {
                        features.push_back( Feature( offset,
                            x,    y, dx, dy*2, -1,
                            x, y+dy, dx, dy,   +2 ) );
                    }
                    // haar_x3
                    if ( (x+dx*3 < W_WIDTH) && (y+dy < W_HEIGHT) )
                    {
                        features.push_back( Feature( offset,
                            x,    y, dx*3, dy, -1,
                            x+dx, y, dx  , dy, +3 ) );
                    }
                    // haar_y3
                    if ( (x+dx < W_WIDTH) && (y+dy*3 < W_HEIGHT) )
                    {
                        features.push_back( Feature( offset,
                            x, y,    dx, dy*3, -1,
                            x, y+dy, dx, dy,   +3 ) );
                    }

                    // x2_y2
                    if ( (x+dx*2 < W_WIDTH) && (y+dy*2 < W_HEIGHT) )
                    {
                        features.push_back( Feature( offset,
                            x,    y,    dx*2, dy*2, -1,
                            x,    y,    dx,   dy,   +2,
                            x+dx, y+dy, dx,   dy,   +2 ) );
                    }
                }
            }
        }
    }
}

cv::Mat_<int> ComputeIntegralImage(const cv::Mat_<int> &mat, int mode) {

	Mat_<int> ii(mat.rows, mat.cols);

	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {

			int p4 = mat(y, x);
			int p3 = (x == 0) ? 0 : ii(y, x - 1);
			int p2 = (y == 0) ? 0 : ii(y - 1, x);
			int p1 = ((x == 0) || (y == 0)) ? 0 : ii(y - 1, x - 1);

			ii(y, x) = ((mode == SQUARED_SUM) ? SQR(p4) : p4) - p1 + p3 + p2;
		}
	}

	return ii;
}
