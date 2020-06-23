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
    // Read dataframe and testframe
    std::ifstream in("train_bank.txt"), test("test_bank.txt");
    list<vector<double>> dataframe, testframe;
    load_csv(in, dataframe);
    load_csv(test, testframe);
    // Print frames and compute a threshold level
    list<vector<double>>::iterator i;
    cout << "DATA FRAME:" << endl << endl;
    for (i = dataframe.begin(); i != dataframe.end(); ++i)
        cout << *i << endl;
    cout << endl << "TEST FRAME:" << endl << endl;
    for (i = testframe.begin(); i != testframe.end(); ++i)
        cout << *i << endl;
    // Train model
    cout << endl << "Training model... " << endl << endl;
    vector<double> model = train(dataframe, .0001);
    list<int> predictions;  // Make predictions
    int guess, zero_correct = 0, zero_incorrect = 0, one_correct = 0, one_incorrect = 0;
    for (i = testframe.begin(); i != testframe.end(); ++i)
    {
        guess = classify(sigmoid(dot_product(model, *i)), .875);
        if (guess == i->back())
        {
            if (guess == 0)
                ++zero_correct;
            else
                ++one_correct;
        }
        else if (guess == 0)
            ++zero_incorrect;
        else
            ++one_incorrect;
    }
    // Print trained model
    cout << endl << "MODEL:" << endl;
    cout << "H(X) = (" << *model.begin() << ")x0";
    int count = 1;
    vector<double>::iterator param = ++model.begin();
    for (vector<double>::iterator param = ++model.begin(); param != model.end(); ++param)
    {
        cout << " + (" << *param << ")x" + std::to_string(count);
        ++count;
    }
    // Report accuracy
    cout << endl << endl << "Points tested: " << testframe.size() << endl;
    cout << "Correct guesses: " << zero_correct + one_correct << ", ";
    cout << (100.0*(zero_correct + one_correct)/testframe.size()) << "%" << endl;
    cout << endl << "0's in dataset: " << zero_correct + one_incorrect << endl;
    cout << "0's guessed: " << zero_incorrect + zero_correct << endl;
    cout << "0's correctly guessed: " << zero_correct << ", ";
    cout << 100.0*zero_correct/(zero_correct + one_incorrect) << "%" << endl;
    cout << endl << "1's in dataset: " << one_correct + zero_incorrect << endl;
    cout << "1's guessed: " << one_correct + one_incorrect << endl;
    cout << "1's correctly guessed: " << one_correct << ", ";
    cout << 100.0*one_correct/(one_correct + zero_incorrect) << "%" << endl << endl;
}
