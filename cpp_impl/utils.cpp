/*
 * utils.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include "utils.h"
#include <stdio.h>
#include <sstream>

void ReadWholeFile(const char *path, char **out_content) {
	FILE *f;
	long file_size;

	if ((f = fopen(path, "rt")) == NULL) {
		printf("NO file\n");
		return;
	}

	fseek(f, 0, SEEK_END);
	file_size = ftell(f);
	fseek(f, 0, SEEK_SET);


	*out_content = new char[file_size];

	fread(*out_content, sizeof(char), file_size, f);
	(*out_content)[file_size - 1] = '\0';
}


std::vector<std::string> &Split(const std::string &s, char delim, std::vector<std::string> &elems)
{
	std::stringstream ss(s);
	std::string item;
    while(std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> Split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    return Split(s, delim, elems);
}
