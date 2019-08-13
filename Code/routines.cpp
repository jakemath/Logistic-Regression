//
//  routines.cpp
//  logistic_regression
//
//  Created by Jacob Mathai on 3/15/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "routines.h"

// Sigmoid function
double sigmoid(double z)
{
    return (1.0 / (1.0 + exp(-1.0 * z)));
}

// Compute vector dot product
double dot_product (vector<double>& v1, vector<double>& v2)
{
    double total = 0.0;
    vector<double>::iterator i = v1.begin(), j = v2.begin();
    for (; i != v1.end() && j != v2.end(); ++i, ++j)
        total += (*i) * (*j);
    return total;
}

// Compute cost of model against train set
double cost(list<vector<double>>& dataset, vector<double>& model)
{
    double cost_total = 0.0;
    double predicted;   // Sum model cost for each data point
    list<vector<double>>::iterator i = dataset.begin(); // Iterate through training set
    for (; i != dataset.end(); ++i)
    {
        predicted = sigmoid(dot_product(model, *i)); // Predicted classification
        cost_total += (-1.0 * i -> back() * log(predicted)) + (1 - i -> back()) * log(1.0 - predicted);
    }
    return -1.0 * cost_total / dataset.size(); // Return cost per point
}

// Train model against train set using gradient descent,
// with given learning rate lr
vector<double> train(list<vector<double>>& dataset, double lr)
{
    vector<double> model(dataset.front().size(), 0.0); // Create 0-filled initial model
    vector<double>::iterator i;
    list<vector<double>>::iterator j;  // Iterate through dataset for each weight
    double gradient;
    double train_cost = cost(dataset, model); // Compute initial cost
    cout << "COST: " << train_cost << endl;
    int component; // Track component
    for (; abs(train_cost) > .001; )    // Iterate to minimal cost
    {
        component = 0; // Compute gradient for each component of model
        i = model.begin();
        for (; i != model.end(); ++i)   // Update model using gradient descent
        {
            gradient = 0.0;
            j = dataset.begin();  // Iterate through dataset for each weight
            for (; j != dataset.end(); ++j)
                gradient += (sigmoid(dot_product(model, *j)) - j -> back()) * (*j)[component];
            *i -= lr * (gradient / dataset.size()); /// dataset.size());    // Update weight
            ++component; // Next component gradient
        }
        train_cost = cost(dataset, model);  // Compute cost for updated model
        cout << "COST: " << train_cost << endl;
    }
    return model;   // Return trained model
}

int classify(double hypothesis, double thresh)
{
    if (hypothesis >= thresh)
        return 1;
    return 0;
}

// Function for parsing comma-separated values in a single line
void split(const string& s, char c, vector<double>& p) 
{
    string str;
    std::istringstream tokenStream (s);
    while (getline(tokenStream, str, ','))
        p.push_back(stod(str));
}

// Function for loading csv file into a list of vectors
void load_csv(std::istream& in, list<vector<double>>& frame)
{
    string str;
    while (!in.eof())
    {
        vector<double> p;
        getline(in, str, '\n');
        split(str, ',', p);
        frame.push_back(p);
    }
}

std::ostream& operator <<(std::ostream& out, const vector<double>& p)
{
    vector<double>::const_iterator i = p.begin();
    for (; i != p.end(); ++i)
        out << *i << " ";
    return out;
}
