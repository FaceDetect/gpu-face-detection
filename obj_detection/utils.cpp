/*
 * utils.cpp
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#include "utils.h"
#include "constants.h"

#include <fstream>
#include <stdexcept>

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

void LoadImages(const char* pos_list, const char* neg_list,
		std::vector<LabeledImg> &container) {

	ifstream pos_imgs(pos_list);
	if (!pos_imgs.is_open()) throw std::runtime_error("Positive images list not found.");

	ifstream neg_imgs(neg_list);
	if (!neg_imgs.is_open()) throw std::runtime_error("Negative images list not found.");

	string img_path;

	while(pos_imgs >> img_path)
		container.push_back(make_pair((Mat_<float>)imread(img_path, CV_LOAD_IMAGE_GRAYSCALE), POSITIVE_LABEL));

	while(neg_imgs >> img_path) {

		Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
		Mat res;
		resize(img, res, Size(W_WIDTH, W_HEIGHT));

		container.push_back(make_pair((Mat_<float>)res, NEGATIVE_LABEL));
	}

	pos_imgs.close();
	neg_imgs.close();

}

std::vector<Feature> &GetFeatureSet() {
	static vector<Feature> set;

	if (set.empty()) GenerateFeatures(set);

	return set;
}
