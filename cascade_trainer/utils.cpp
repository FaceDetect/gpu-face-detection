/*
 * utils.cpp
 *
 *  Created on: Feb 24, 2013
 *      Author: olehp
 */

#include "utils.h"

using namespace std;


void GenerateFeatures(std::vector<Feature>& features)
{
    int offset = W_WIDTH + 1;
    for( int x = 0; x < W_WIDTH; x++ )
    {
        for( int y = 0; y < W_HEIGHT; y++ )
        {
            for( int dx = 1; dx <= W_WIDTH; dx++ )
            {
                for( int dy = 1; dy <= W_HEIGHT; dy++ )
                {
                    // haar_x2
                    if ( (x+dx*2 <= W_WIDTH) && (y+dy <= W_HEIGHT) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y, dx*2, dy, -1,
                            x+dx, y, dx  , dy, +2 ) );
                    }
                    // haar_y2
                    if ( (x+dx <= W_WIDTH) && (y+dy*2 <= W_HEIGHT) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y, dx, dy*2, -1,
                            x, y+dy, dx, dy,   +2 ) );
                    }
                    // haar_x3
                    if ( (x+dx*3 <= W_WIDTH) && (y+dy <= W_HEIGHT) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y, dx*3, dy, -1,
                            x+dx, y, dx  , dy, +3 ) );
                    }
                    // haar_y3
                    if ( (x+dx <= W_WIDTH) && (y+dy*3 <= W_HEIGHT) )
                    {
                        features.push_back( Feature( offset, false,
                            x, y,    dx, dy*3, -1,
                            x, y+dy, dx, dy,   +3 ) );
                    }

                    // x2_y2
                    if ( (x+dx*2 <= W_WIDTH) && (y+dy*2 <= W_HEIGHT) )
                    {
                        features.push_back( Feature( offset, false,
                            x,    y,    dx*2, dy*2, -1,
                            x,    y,    dx,   dy,   +2,
                            x+dx, y+dy, dx,   dy,   +2 ) );
                    }
                }
            }
        }
    }
}

