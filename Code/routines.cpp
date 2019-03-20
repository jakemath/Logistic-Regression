//
//  routines.cpp
//  logistic_regression
//
//  Created by Jacob Mathai on 3/15/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "routines.h"

double sigmoid (double z) { return (1 / (1 + exp(-1 * z))); }

double hypothesis (double z) { return sigmoid(z); }

double dot_product (vector<double>& model, vector<double>& point)
{
    double total = 0;
    vector<double>::iterator i = model.begin(), j = point.begin();
    for (; i != model.end() && j != point.end(); ++i, ++j)
        total += (*i) * (*j);
    return total;
}

double cost (list<point>& train_set, vector<double>& model)
{
    list<point>::iterator i = train_set.begin();    // Iterate through training set
    double cost_total = 0, actual, predicted;   // Sum model cost for each data point
    for (; i != train_set.end(); ++i)
    {
        actual = i -> attributes.back();
        predicted = hypothesis (dot_product(model, i -> attributes));
        cost_total += ((-1 * actual * log(predicted)) - ((1 - actual) * log(1 - predicted)));
    }
    return (1.0 * cost_total / train_set.size());   // Return average cost
}

vector<double> train (list<point>& dataset, double lr)
{
    vector<double> model (dataset.front().attributes.size(), 0.0);    // Create 0-filled initial model
    vector<double>::iterator i = model.begin(), x = dataset.front().attributes.begin();
    double train_cost = cost (dataset, model);
    while (train_cost > .01)    // Iterate until cost is minimized
    {
        int component = 0;
        for (; i != model.end(); ++i)   // Update model using gradient descent
        {
            double gradient = 0, hypothesized, actual;    // Sum
            list<point>::iterator j = dataset.begin();  // Iterate through dataset for each weight
            x = dataset.front().attributes.begin();
            for (; j != dataset.end(); ++j)
            {
                hypothesized = hypothesis(dot_product(model, j -> attributes));
                actual = j -> attributes.back();
                gradient += (hypothesized - actual) * (x[component]);
            }
            *i -= lr * gradient;    // Update weight
            ++component;
        }
        train_cost = cost (dataset, model);  // Compute cost for updated model
        i = model.begin();  // Re-iterate
    }
    return model;   // Return trained model
}

int classify (double hypothesis, double thresh)
{
    if (hypothesis >= thresh)
        return 1;
    return 0;
}

int predict (vector<double>& model, point p)
{
    double hypothesized = hypothesis(dot_product(model, p.attributes));
    return (classify(hypothesized, 0.90));
}

void split (const string& s, char c, point& p)  // Function for parsing comma separated values in a line of the txt file into a point instance
{
    string str;
    std::istringstream tokenStream (s);
    while (getline(tokenStream, str, ','))
        p.attributes.push_back(stod(str));
}

void load_csv (std::istream& in, list<point>& frame)  // Function for loading comma separated values in txt file into a vector of points
{
    string str;
    while (!in.eof())
    {
        point p;
        getline(in, str, '\n');
        split(str,',',p);
        frame.push_back(p);
    }
}

std::ostream& operator << (std::ostream& out, const point& p)
{
    vector<double>::const_iterator i = p.attributes.begin();
    for (; i != p.attributes.end(); ++i)
        out << *i << " ";
    return out;
}
