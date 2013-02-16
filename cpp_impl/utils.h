/*
 * utils.h
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <string>

#define MAX(a,b) (( a ) >= ( b ) ? ( a ) : (  b ))
#define MIN(a,b) (( a ) < ( b ) ? ( a ) : ( b ))
#define SQR(a) (( a ) * ( a ))

std::vector<std::string> &Split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> Split(const std::string &s, char delim);
void ReadWholeFile(const char *path, char **out_content);

#endif /* UTILS_H_ */
