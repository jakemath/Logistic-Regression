//
//  routines.h
//  logistic_regression
//
//  Created by Jacob Mathai on 3/15/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#ifndef routines_h
#define routines_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <vector>
#include <iterator>
#include <string>
#include <cmath>

using std::list;
using std::vector;
using std::string;
using std::cout;
using std::endl;

struct point // Data point struct for parsing data, values stored in list attributes
{
    point () {}
    vector<double> attributes;
};

double sigmoid (double z);  // Sigmoid function
double dot_product (vector<double>& model, vector<double>& point); // Vector dot product function
double hypothesis (double z);   // Hypothesis function
double cost (list<point>& train_set, vector<double>& model); // Cost function
vector<double> train (list<point>& dataset, double lr); // Model training function using gradient descent

int classify (double hypothesis, double thresh); // Classify hypothesized value based on threshold
int predict (vector<double>& model, point p);

void split (const string& s, char c, point& p);
void load_csv (std::istream& in, list<point>& frame); // Function to create list of data points from dataset

std::ostream& operator << (std::ostream& out, const point& p);  // Print point

#endif /* routines_h */
