g++ -std=c++2a -c routines.cpp routines.h
g++ -std=c++2a -c main.cpp
g++ -o run main.o routines.o
