#include "mnist.h"

//Loads all 60000 training images and all 10000 testing images.

void deinit() {
	free_mnist();
}
void load() {
    load_mnist();
	/*
    int i;
    for (i=0; i<784; i++) {
        printf("%1.1f ", test_image[0][i]);
        if ((i+1) % 28 == 0) putchar('\n');
    }
	*/
}

//Images are 28x28px in size.
void get_test_images(double*** ptr, int* n_img, int* img_size) {
    (*ptr) = test_image;
    *n_img = NUM_TEST;
    if(img_size != NULL) {
        *img_size = SIZE;
    }
}

void get_train_images(double*** ptr, int* n_img, int* img_size) {
    *ptr = train_image;
    *n_img = NUM_TRAIN;
    if (img_size != NULL) {
        *img_size = SIZE;
    }
}

void get_test_labels(int** ptr, int* n_labels) {
    *ptr = test_label;
    *n_labels = NUM_TEST;
}

void get_train_labels(int** ptr, int* n_labels) {
    *ptr = train_label;
    *n_labels = NUM_TRAIN;
}
