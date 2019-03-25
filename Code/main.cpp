//
//  main.cpp
//  logistic_regression
//
//  Created by Jacob Mathai on 3/15/19.
//  Copyright © 2019 Jacob Mathai. All rights reserved.
//

#include "routines.h"

int main(int argc, char* argv[])
{
    std::ifstream in("train_bank.txt"), test("test_bank.txt"); // Read dataframe and testframe
    list<point> dataframe, testframe;   // Store data points in a list
    load_csv(in, dataframe);    // Each point contains a vector of feature values
    load_csv(test, testframe);
    
    list<point>::iterator i = dataframe.begin();
    cout << "DATA FRAME:" << endl << endl;  // Print frames
    for (; i != dataframe.end(); ++i)
        cout << *i << endl;
    
    i = testframe.begin();
    cout << endl << "TEST FRAME:" << endl << endl;
    for (; i != testframe.end(); ++i)
        cout << *i << endl;
    
    vector<double> model = train (dataframe, 0.0005); // Train model
    list<int> predictions;  // Make predictions
    i = testframe.begin();
    for (; i != testframe.end(); ++i)
        predictions.push_back(predict(model, *i));
    
    i = testframe.begin();  // Compute accuracy
    list<int>::iterator j = predictions.begin();
    int count = 0, zero_count = 0, one_count = 0;
    for (; i != testframe.end() && j != predictions.end(); ++i, ++j)
    {
        if (*j == i -> attributes.back())
        {
            count += 1;
            if (*j == 0)
                zero_count += 1;
            else
                one_count += 1;
        }
    }
    
    cout << endl << "Predicted - Actual" << endl << endl;    // Print predictions
    i = testframe.begin();
    j = predictions.begin();
    for (; i != testframe.end() && j != predictions.end(); ++i, ++j)
        cout << "\t" << *j << "\t" << i -> attributes.back() << endl;
    
    cout << endl << "MODEL PARAMETERS:" << endl; // Print trained model
    vector<double>::iterator m = model.begin();
    for (; m != model.end(); ++m)
        cout << *m << " ";
    
    cout << endl << endl << "0's correctly guessed: " << zero_count << endl;
    cout << "1's correctly guessed: " << one_count << endl << "Accuracy: ";
    cout << (1.0 * count / predictions.size()) * 100 << "%" << endl << endl;
}
