//A wrapper arround some helper functions for loading the mnist dataset written in ancient c.
//The original header file does not compile on modern c++ compilers.
//You need to use a c compiler (CMAKE should take care of that)
#pragma once

void load();
void deinit();

void get_test_images(double*** ptr, int* n_img, int* img_size);
void get_train_images(double*** ptr, int* n_img, int* img_size);
void get_test_labels(int** ptr, int* n_labels);
void get_train_labels(int** ptr, int* n_labels);
