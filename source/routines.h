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

double sigmoid(double z);
double dot_product(vector<double>& v1, vector<double>& v2); // Vector dot product function
int classify(double hypothesis, double thresh); // Classify sigmoid value based on threshold
double cost(list<vector<double>>& dataset, vector<double>& model); // Cost function
vector<double> train(list<vector<double>>& dataset, double lr); // Model training function
void split(const string& s, char c, vector<double>& p);
void load_csv(std::istream& in, list<vector<double>>& frame); // Function to create list of data points from dataset
std::ostream& operator << (std::ostream& out, const vector<double>& p);  // Print point

#endif /* routines_h */
