/*
 * main.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: olehp
 */

#include <stdio.h>
#include "ObjectRecognizer.h"


int main(int argv, char **args)
{
	ObjectRecognizer obj_rec;

	obj_rec.LoadHaarCascade("../data/haarcascade_frontalface_alt.xml");
	obj_rec.LoadImage("../data/lena.jpg");


}
