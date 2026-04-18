#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace chrono;

// Function to calculate RMS of an RGB pixel
double rms(const Vec3b& pixel) {
    return sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1] + pixel[2] * pixel[2]);
}

int colorNoise(int color) {
    int delta = 30;
    return color + rand() % min(255 - color, delta + 1);
}

void noise(Vec3b& pixel) {
    pixel[0] = colorNoise(pixel[0]);
    pixel[1] = colorNoise(pixel[1]);
    pixel[2] = colorNoise(pixel[2]);
}

int pullPixel(int v1, int v2) {
    int avg = (v1 + v2) / 2;
    int c1 = v1 - avg;
    int c2 = v2 - avg;
    int negDelta = min(c1, c2);
    int posDelta = (negDelta == c2) ? c1 : c2;
    return avg + (rand() % (posDelta - negDelta + 1) + negDelta);
}

void pull(Vec3b& pixel, const Vec3b& pull_pixel) {
    pixel[0] = pullPixel(pixel[0], pull_pixel[0]);
    pixel[1] = pullPixel(pixel[1], pull_pixel[1]);
    pixel[2] = pullPixel(pixel[2], pull_pixel[2]);
}

void process_iteration(Mat& pixels, int width, int height, bool normal, const Mat &count) {
    Mat temp = pixels.clone();
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            Vec3b center = temp.at<Vec3b>(j, i);
            int n = count.at<int>(j, i);
            if (normal) {
                pixels.at<Vec3b>(j, i)[0] = center[0] / n;
                pixels.at<Vec3b>(j, i)[1] = center[1] / n;
                pixels.at<Vec3b>(j, i)[2] = center[2] / n;
            } else {
                pixels.at<Vec3b>(j, i) = center;
            }
        }
    }
    for (int j = 0; j < height; j++) {
        for (int i = 1; i < width; i++) {
            Vec3b left = temp.at<Vec3b>(j, i - 1);
            int n = count.at<int>(j, i);
            pixels.at<Vec3b>(j, i)[0] += left[0] / n;
            pixels.at<Vec3b>(j, i)[1] += left[1] / n;
            pixels.at<Vec3b>(j, i)[2] += left[2] / n;
        }
    }
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width - 1; i++) {
            Vec3b right = temp.at<Vec3b>(j, i + 1);
            int n = count.at<int>(j, i);
            pixels.at<Vec3b>(j, i)[0] += right[0] / n;
            pixels.at<Vec3b>(j, i)[1] += right[1] / n;
            pixels.at<Vec3b>(j, i)[2] += right[2] / n;
        }
    }
    for (int j = 1; j < height; j++) {
        for (int i = 0; i < width; i++) {
            Vec3b top = temp.at<Vec3b>(j - 1, i);
            int n = count.at<int>(j, i);
            pixels.at<Vec3b>(j, i)[0] += top[0] / n;
            pixels.at<Vec3b>(j, i)[1] += top[1] / n;
            pixels.at<Vec3b>(j, i)[2] += top[2] / n;
        }
    }
    for (int j = 0; j < height - 1; j++) {
        for (int i = 0; i < width; i++) {
            Vec3b bottom = temp.at<Vec3b>(j + 1, i);
            int n = count.at<int>(j, i);
            pixels.at<Vec3b>(j, i)[0] += bottom[0] / n;
            pixels.at<Vec3b>(j, i)[1] += bottom[1] / n;
            pixels.at<Vec3b>(j, i)[2] += bottom[2] / n;
        }
    }
}

void do_blende(Mat& pixels, int width, int height, int num_it, const string& type_blende) {
    bool normal = (type_blende == "normal");
    if (!normal && type_blende != "brighten") return;
    Mat count = Mat::zeros(pixels.size(), CV_32S);
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            count.at<int>(j, i) = 1 + (i > 1) + (i < width) + (j > 1) + (j < height);
        }
    }
    for (int it = 0; it < num_it; it++) {
        process_iteration(pixels, width, height, normal, count);
    }
}

void fbm_colorize(Mat& pixels, int width, int height, int num_it, int octaves, double persistence, double rms_l, double rms_u, const Vec3b& target_color, double percent_chance) {
    const double MAX_RMS = sqrt(3.0 * 255.0 * 255.0);
    int MARKER_CHANCE = int(percent_chance * 100);
    for (int n = 0; n < num_it; ++n) {
        Mat temp = pixels.clone();
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                Vec3b &px = pixels.at<Vec3b>(j, i);
                double cur_rms = rms(px);
                if (cur_rms >= rms_l && cur_rms <= rms_u) {
                    double sum = 0.0, amp = 1.0, freq = 1.0, maxAmp = 0.0;
                    for (int o = 0; o < octaves; ++o) {
                        int xi = int(i * freq) % width;
                        int yj = int(j * freq) % height;
                        Vec3b &sample = temp.at<Vec3b>(yj, xi);
                        double v = rms(sample) / MAX_RMS;
                        sum += amp * v;
                        maxAmp += amp;
                        amp *= persistence;
                        freq *= 2.0;
                    }
                    double n = sum / maxAmp;
                    if (rand() % 100 < MARKER_CHANCE) {
                        for (int c = 0; c < 3; ++c) {
                            double blended = px[c] + (target_color[c] - px[c]) * n;
                            px[c] = saturate_cast<uchar>(blended);
                        }
                    }
                }
            }
        }
    }
}

void perform_filter(Mat& image, int width, int height) {
    for (int i = 0; i < 3; ++i) {
        auto start = high_resolution_clock::now();
        fbm_colorize(image, width, height, 3, 3, 0.55, 0.0, 90.0, {70, 50, 120}, 0.40);
        fbm_colorize(image, width, height, 3, 6, 0.95, 50.0, 160.0, {60, 90, 70}, 0.38);
        fbm_colorize(image, width, height, 3, 3, 0.15, 130.0, 250.0, {140, 50, 110}, 0.35);
        fbm_colorize(image, width, height, 3, 6, 0.95, 220.0, 360.0, {170, 100, 130}, 0.30);
        do_blende(image, width, height, 5, "normal");
        do_blende(image, width, height, 5, "normal");
        cout << i << " Took....";
        cout << (duration_cast<microseconds>(high_resolution_clock::now() - start)).count() * pow(10, -6) / 60 << endl;
    }
}

int main() {
    string im_name = "cigs.";
    string tag = ".png";
    string version = "1";
    int frame_count = 0;
    Mat image = imread(im_name + tag, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return 1;
    }
    int width = image.cols;
    int height = image.rows;
    perform_filter(image, width, height);
    imwrite(im_name + "_" + version + ".png", image);
    return 0;
}


/*
clang++ -std=c++11 -O3 -march=native -flto -funroll-loops -ffast-math -Rpass=loop-vectorize -fopenmp -L/opt/homebrew/lib -I/opt/homebrew/include -lomp other_2.cpp -o output `pkg-config --cflags --libs opencv4`


*/