/*
 * main.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include <stdio.h>
#include "ObjectRecognizer.h"
#include <iostream>

int main(int argv, char **args)
{
	ObjectRecognizer obj_rec;
	std::cout << args[1] << " ";
	obj_rec.LoadHaarCascade(args[2]);
	obj_rec.LoadImage(args[1]);
	obj_rec.Recognize();


}
