//
//  routines.cpp
//  logistic_regression
//
//  Created by Jacob Mathai on 3/15/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "routines.h"

double sigmoid (double z) { return (1.0 / (1.0 + exp(-1.0 * z))); }

double hypothesis (double z) { return sigmoid(z); }

double dot_product (vector<double>& model, vector<double>& point)
{
    double total = 0.0;
    vector<double>::iterator i = model.begin(), j = point.begin();
    for (; i != model.end() && j != point.end(); ++i, ++j)
        total += (*i) * (*j);
    return total;
}

double cost (list<point>& train_set, vector<double>& model)
{
    list<point>::iterator i = train_set.begin();    // Iterate through training set
    double cost_total = 0.0, actual, predicted;   // Sum model cost for each data point
    for (; i != train_set.end(); ++i)
    {
        actual = i -> attributes.back();
        predicted = hypothesis(dot_product(model, i -> attributes));
        cost_total += ((-1.0 * actual * log(predicted)) - ((1.0 - actual) * log(1.0 - predicted)));
    }
    return (1.0 * cost_total / train_set.size());   // Return average cost
}

vector<double> train (list<point>& dataset, double lr)
{
    vector<double> model(dataset.front().attributes.size(), 0.0);    // Create 0-filled initial model
    vector<double>::iterator i = model.begin();
    list<point>::iterator j;  // Iterate through dataset for each weight;
    double train_cost = cost(dataset, model);
    int component;
    double gradient, hypothesized, actual;
    while (train_cost > .0000001)    // Iterate until cost is minimized
    {
        component = 0;
        for (; i != model.end(); ++i)   // Update model using gradient descent
        {
            gradient = 0.0;
            j = dataset.begin();  // Iterate through dataset for each weight
            for (; j != dataset.end(); ++j)
            {
                hypothesized = hypothesis(dot_product(model, j -> attributes));
                actual = j -> attributes.back();
                gradient += (hypothesized - actual) * j -> attributes[component];
            }
            *i -= lr * gradient;    // Update weight
            ++component;
        }
        train_cost = cost(dataset, model);  // Compute cost for updated model
        cout << "COST: " << train_cost << endl;
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

int predict (vector<double>& model, point p) { return classify(hypothesis(dot_product(model, p.attributes)), 0.996); }

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
