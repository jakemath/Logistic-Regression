//
//  routines.cpp
//  logistic_regression
//
//  Created by Jacob Mathai on 3/15/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "routines.h"

// Sigmoid function
double sigmoid(double z) { return (1.0/(1.0 + exp(-1.0*z))); }

// Compute cost of model against train set
double cost(list<vector<double>>& dataset, vector<double>& model) {
    double predicted, cost_total = 0.0;
    for (list<vector<double>>::iterator datapoint = dataset.begin(); i != dataset.end(); ++i) {
        predicted = sigmoid(std::inner_product(model.begin(), i->begin(), model.end(), 0.0));
        cost_total += (-1.0*i->back()*log(predicted)) + (1 - i->back())*log(1.0 - predicted);
    }
    return -1.0*cost_total/dataset.size();
}

// Train model against train set using gradient descent, with given learning rate lr
vector<double> train(list<vector<double>>& dataset, double lr) {
    vector<double> model(dataset.front().size(), 0.0);
    vector<double>::iterator param;
    list<vector<double>>::iterator datapoint;
    int component;
    double gradient, dot_product, train_cost = cost(dataset, model);
    cout << "COST: " << train_cost << endl;
    while (abs(train_cost) > .001) {
        component = 0;
        for (param = model.begin(); param != model.end(); ++param) {
            gradient = 0.0;
            for (datapoint = dataset.begin(); datapoint != dataset.end(); ++datapoint) {
                dot_product = std::inner_product(model.begin(), datapoint->begin(), model.end(), 0.0)
                gradient += (sigmoid(dot_product) - datapoint->back())*(*datapoint)[component];
            }
            *param -= lr*gradient/dataset.size();
            ++component;
        }
        train_cost = cost(dataset, model);
        cout << "COST: " << train_cost << endl;
    }
    return model;
}

int classify(double hypothesis, double thresh) { return hypothesis >= thresh; }

// Function for parsing comma-separated values in a single line
void split(const string& s, char c, vector<double>& p) {
    string str;
    std::istringstream tokenStream (s);
    while (getline(tokenStream, str, ','))
        p.push_back(stod(str));
}

// Function for loading csv file into a list of vectors
void load_csv(std::istream& in, list<vector<double>>& frame) {
    string str;
    while (!in.eof()) {
        vector<double> p;
        getline(in, str, '\n');
        split(str, ',', p);
        frame.push_back(p);
    }
}

std::ostream& operator <<(std::ostream& out, const vector<double>& p) {
    for (vector<double>::const_iterator i = p.begin(); i != p.end(); ++i)
        out << *i << " ";
    return out;
}
