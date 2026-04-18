#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <opencv2/opencv.hpp> 
#include <string>
#include <stdexcept>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace chrono;

struct RunProfile {
    string shading;
    int base_passes;
};

struct RunSpec {
    string label;
    double intensity;
    double drift;
};

struct RandomBand {
    double rms_low;
    double rms_high;
    Vec3b target_color;
    double mix_scale;
    int passes;
    string mix_pattern;
    string shift_pattern_x;
    string shift_pattern_y;
    int shift_radius;
};

namespace term {
    const string reset = "\033[0m";
    const string bold = "\033[1m";
    const string dim = "\033[2m";
    const string cyan = "\033[36m";
    const string blue = "\033[34m";
    const string green = "\033[32m";
    const string yellow = "\033[33m";
    const string magenta = "\033[35m";
    const string red = "\033[31m";
}

string format_seconds(long long microseconds_total) {
    ostringstream out;
    out.setf(ios::fixed);
    out.precision(2);
    out << (microseconds_total / 1000000.0);
    return out.str();
}

string format_double(double value, int precision = 2) {
    ostringstream out;
    out.setf(ios::fixed);
    out.precision(precision);
    out << value;
    return out.str();
}

void print_divider(const string& color = term::dim) {
    cout << color << "------------------------------------------------------------" << term::reset << endl;
}

void print_run_plan(const vector<RunSpec>& runs) {
    print_divider(term::blue);
    cout << term::bold << term::cyan << "Discrete Runner" << term::reset
         << term::dim << "  parsed runs.txt and built the render plan" << term::reset << endl;
    print_divider(term::blue);
    for (size_t i = 0; i < runs.size(); ++i) {
        const RunSpec& run = runs[i];
        cout << term::magenta << "  [" << (i + 1) << "] " << term::reset
             << term::bold << run.label << term::reset
             << term::dim << "  bands=random(3-4)"
             << "  intensity=" << format_double(run.intensity)
             << "  drift=" << format_double(run.drift)
             << term::reset << endl;
    }
    print_divider(term::blue);
}

void print_source_plan(const vector<string>& files, const string& workspace) {
    cout << term::bold << term::cyan << "Input Scan" << term::reset
         << term::dim << "  source PNGs found in ./" << workspace << term::reset << endl;
    for (size_t i = 0; i < files.size(); ++i) {
        cout << term::green << "  [" << (i + 1) << "] " << term::reset << files[i] << endl;
    }
    print_divider();
}

string trim(const string& value) {
    const string whitespace = " \t\r\n";
    size_t start = value.find_first_not_of(whitespace);
    if (start == string::npos) {
        return "";
    }
    size_t end = value.find_last_not_of(whitespace);
    return value.substr(start, end - start + 1);
}

RunProfile make_run_profile(const string& shading) {
    if (shading == "default") {
        return {"default", 8};
    }
    if (shading == "moody") {
        return {"moody", 10};
    }
    if (shading == "soft") {
        return {"soft", 6};
    }
    if (shading == "electric") {
        return {"electric", 9};
    }
    if (shading == "wash") {
        return {"wash", 7};
    }
    throw invalid_argument("Unknown shading profile: " + shading);
}

double clamp_double(double value, double lower, double upper) {
    return max(lower, min(value, upper));
}

int scaled_passes(int base_passes, double intensity) {
    return max(1, static_cast<int>(round(base_passes * intensity)));
}

vector<string> available_shadings() {
    return {"default", "moody", "soft", "electric", "wash"};
}

RunSpec make_default_run_spec() {
    RunSpec run;
    run.label = "default_mix";
    run.intensity = 1.0;
    run.drift = 1.0;
    return run;
}

vector<RunSpec> read_runs_file(const string& path) {
    ifstream file(path.c_str());
    vector<RunSpec> runs;

    if (!file.is_open()) {
        runs.push_back(make_default_run_spec());
        return runs;
    }

    string line;
    while (getline(file, line)) {
        string cleaned = trim(line);
        if (cleaned.empty() || cleaned[0] == '#') {
            continue;
        }

        istringstream stream(cleaned);
        string label;
        double intensity = 1.0;
        double drift = 1.0;
        stream >> label;
        if (label.empty()) {
            continue;
        }
        if (!(stream >> intensity)) {
            intensity = 1.0;
        }
        if (!(stream >> drift)) {
            drift = 1.0;
        }

        RunSpec run;
        run.label = label;
        run.intensity = clamp_double(intensity, 0.35, 2.5);
        run.drift = clamp_double(drift, 0.25, 2.5);
        runs.push_back(run);
    }

    if (runs.empty()) {
        runs.push_back(make_default_run_spec());
    }

    return runs;
}

// Function to calculate RMS of an RGB pixel
double rms(const Vec3b& pixel) {
    return sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1] + pixel[2] * pixel[2]);
}

// Function to add noise to an individual color value
int colorNoise(int color) {
    int delta = 30;
    return color + rand() % min(255 - color, delta + 1);  // Random number between 0 and delta
}

// Function to add noise to an entire RGB pixel
void noise(Vec3b& pixel) {
    pixel[0] = colorNoise(pixel[0]);
    pixel[1] = colorNoise(pixel[1]);
    pixel[2] = colorNoise(pixel[2]);
}

// Function to pull pixel colors towards a specific color (v1, v2 are individual channel values)
int pullPixel(int v1, int v2) {
    int avg = (v1 + v2) / 2;
    int c1 = v1 - avg;
    int c2 = v2 - avg;
    int negDelta = min(c1, c2);
    int posDelta = (negDelta == c2) ? c1 : c2;
    return avg + (rand() % (posDelta - negDelta + 1) + negDelta);
}

// Blending two pixels
void pull(Vec3b& pixel, const Vec3b& pull_pixel) {
    pixel[0] = pullPixel(pixel[0], pull_pixel[0]);
    pixel[1] = pullPixel(pixel[1], pull_pixel[1]);
    pixel[2] = pullPixel(pixel[2], pull_pixel[2]);
}

double frac(int numerator, int denominator) {
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

struct BlendTap {
    int dx;
    int dy;
    double weight;
};

struct BlendRecipe {
    string name;
    vector<BlendTap> taps;
};

struct FractionPattern {
    string name;
    vector<double> levels;
    vector<vector<int>> grid;
};

BlendRecipe make_blend_recipe(const string& name) {
    if (name == "cross_stitch") {
        return {
            name,
            {
                {-1,  0, frac(1, 8)}, {1,  0, frac(1, 8)},
                { 0, -1, frac(1, 8)}, {0,  1, frac(1, 8)},
                {-1, -1, frac(1, 16)}, {1, -1, frac(1, 16)},
                {-1,  1, frac(1, 16)}, {1,  1, frac(1, 16)}
            }
        };
    }

    if (name == "north_pull") {
        return {
            name,
            {
                { 0, -1, frac(1, 4)},
                {-1, -1, frac(1, 8)}, {1, -1, frac(1, 8)},
                {-1,  0, frac(1, 16)}, {1,  0, frac(1, 16)},
                { 0,  1, frac(1, 16)}
            }
        };
    }

    if (name == "scanline") {
        return {
            name,
            {
                {-1,  0, frac(1, 4)}, {1,  0, frac(1, 4)},
                { 0, -1, frac(1, 16)}, {0,  1, frac(1, 16)},
                {-2,  0, frac(1, 16)}, {2,  0, frac(1, 16)}
            }
        };
    }

    if (name == "halo") {
        return {
            name,
            {
                {-1, -1, frac(1, 8)}, {1, -1, frac(1, 8)},
                {-1,  1, frac(1, 8)}, {1,  1, frac(1, 8)},
                {-1,  0, frac(1, 16)}, {1,  0, frac(1, 16)},
                { 0, -1, frac(1, 16)}, {0,  1, frac(1, 16)}
            }
        };
    }

    if (name == "stutter_step") {
        return {
            name,
            {
                { 1,  0, frac(1, 4)}, {1,  1, frac(1, 8)},
                { 0,  1, frac(1, 8)}, {-1,  0, frac(1, 16)},
                { 0, -1, frac(1, 16)}, {-1, -1, frac(1, 16)}
            }
        };
    }

    throw invalid_argument("Unknown blend recipe: " + name);
}

FractionPattern make_fraction_pattern(const string& name) {
    if (name == "checker_ladder") {
        return {
            name,
            {0.0, frac(1, 8), frac(1, 4), frac(3, 8), frac(1, 2), frac(5, 8)},
            {
                {0, 2, 1, 3},
                {4, 1, 5, 2},
                {1, 3, 0, 4},
                {5, 2, 4, 1}
            }
        };
    }

    if (name == "corner_bloom") {
        return {
            name,
            {0.0, frac(1, 6), frac(1, 3), frac(1, 2), frac(2, 3), frac(5, 6)},
            {
                {5, 3, 2, 4},
                {3, 1, 0, 2},
                {2, 0, 1, 3},
                {4, 2, 3, 5}
            }
        };
    }

    if (name == "scan_bars") {
        return {
            name,
            {0.0, frac(1, 5), frac(2, 5), frac(3, 5), frac(4, 5)},
            {
                {1, 3, 4, 2, 0, 2},
                {1, 3, 4, 2, 0, 2},
                {0, 2, 3, 4, 1, 3},
                {0, 2, 3, 4, 1, 3}
            }
        };
    }

    throw invalid_argument("Unknown fraction pattern: " + name);
}


/*

        BLENDESSS

*/
void process_recipe_iteration(Mat& pixels, const BlendRecipe& recipe) {
    Mat temp = pixels.clone();
    int height = pixels.rows;
    int width = pixels.cols;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            Vec3d blended(0.0, 0.0, 0.0);
            double carried_weight = 1.0;

            for (const BlendTap& tap : recipe.taps) {
                int x = i + tap.dx;
                int y = j + tap.dy;
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    const Vec3b& sample = temp.at<Vec3b>(y, x);
                    blended[0] += tap.weight * sample[0];
                    blended[1] += tap.weight * sample[1];
                    blended[2] += tap.weight * sample[2];
                    carried_weight -= tap.weight;
                }
            }

            const Vec3b& center = temp.at<Vec3b>(j, i);
            blended[0] += max(0.0, carried_weight) * center[0];
            blended[1] += max(0.0, carried_weight) * center[1];
            blended[2] += max(0.0, carried_weight) * center[2];

            pixels.at<Vec3b>(j, i)[0] = saturate_cast<uchar>(blended[0]);
            pixels.at<Vec3b>(j, i)[1] = saturate_cast<uchar>(blended[1]);
            pixels.at<Vec3b>(j, i)[2] = saturate_cast<uchar>(blended[2]);
        }
    }
}

void apply_blend_recipe(Mat& pixels, int num_it, const string& recipe_name) {
    BlendRecipe recipe = make_blend_recipe(recipe_name);
    for (int it = 0; it < num_it; ++it) {
        process_recipe_iteration(pixels, recipe);
    }
}

// Perform distortion on image
void distort_image(Mat& pixels, int width, int height, int num_it, double rms_l, double rms_u, const Vec3b& pull_color, double percent_chance) {
    double cur_rms;
    const double MARKER_CHANCE = percent_chance * 100;
    for (int it = 0; it < num_it; it++) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                // Apply the distortion logic based on band calculator and RMS range
                cur_rms = rms(pixels.at<Vec3b>(j, i));
                if (cur_rms >= rms_l && cur_rms <= rms_u) {
                    if (rand() % 100 < MARKER_CHANCE) {
                        pull(pixels.at<Vec3b>(j, i), pull_color);
                    }
                }
            }
        }
    }
}

double pattern_fraction_at(const FractionPattern& pattern, int x, int y) {
    int pattern_height = pattern.grid.size();
    int pattern_width = pattern.grid[0].size();
    int level_index = pattern.grid[y % pattern_height][x % pattern_width];
    return pattern.levels[level_index];
}

void pattern_colorize(Mat& pixels,
    int width, int height,
    int num_it,
    double rms_l,
    double rms_u,
    const Vec3b& target_color,
    double percent_chance,
    const string& pattern_name,
    int x_offset = 0,
    int y_offset = 0)
{
    FractionPattern pattern = make_fraction_pattern(pattern_name);
    int marker_chance = int(percent_chance * 100);

    for (int n = 0; n < num_it; ++n) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                Vec3b& px = pixels.at<Vec3b>(j, i);
                double cur_rms = rms(px);
                if (cur_rms >= rms_l && cur_rms <= rms_u && rand() % 100 < marker_chance) {
                    double mix = pattern_fraction_at(pattern, i + x_offset + n, j + y_offset + n);
                    for (int c = 0; c < 3; ++c) {
                        double blended = px[c] + (target_color[c] - px[c]) * mix;
                        px[c] = saturate_cast<uchar>(blended);
                    }
                }
            }
        }
    }
}

void shift_colorize(Mat& pixels,
    int width, int height,
    int num_it,
    double rms_l,
    double rms_u,
    const Vec3b& target_color,
    double mix_scale,
    const string& mix_pattern_name,
    const string& shift_pattern_x_name,
    const string& shift_pattern_y_name,
    int shift_radius,
    int x_offset = 0,
    int y_offset = 0)
{
    FractionPattern mix_pattern = make_fraction_pattern(mix_pattern_name);
    FractionPattern shift_pattern_x = make_fraction_pattern(shift_pattern_x_name);
    FractionPattern shift_pattern_y = make_fraction_pattern(shift_pattern_y_name);

    for (int n = 0; n < num_it; ++n) {
        Mat temp = pixels.clone();
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                Vec3b& px = pixels.at<Vec3b>(j, i);
                double cur_rms = rms(px);
                if (cur_rms < rms_l || cur_rms > rms_u) {
                    continue;
                }

                double mix = clamp_double(pattern_fraction_at(mix_pattern, i + x_offset + n, j + y_offset + n) * mix_scale, 0.0, 1.0);
                int shift_x = static_cast<int>(round((pattern_fraction_at(shift_pattern_x, i + x_offset + n, j + y_offset + n) - 0.5) * 2.0 * shift_radius));
                int shift_y = static_cast<int>(round((pattern_fraction_at(shift_pattern_y, i + x_offset + n, j + y_offset + n) - 0.5) * 2.0 * shift_radius));
                int sx = max(0, min(width - 1, i + shift_x));
                int sy = max(0, min(height - 1, j + shift_y));
                const Vec3b& shifted = temp.at<Vec3b>(sy, sx);

                for (int c = 0; c < 3; ++c) {
                    double moved = shifted[c] + (target_color[c] - shifted[c]) * mix;
                    px[c] = saturate_cast<uchar>(moved);
                }
            }
        }
    }
}

string random_pattern_name() {
    const string patterns[] = {"checker_ladder", "corner_bloom", "scan_bars"};
    return patterns[rand() % 3];
}

Vec3b random_awkward_color() {
    const Vec3b palette[] = {
        Vec3b(15, 210, 250),
        Vec3b(240, 40, 85),
        Vec3b(35, 250, 145),
        Vec3b(250, 205, 35),
        Vec3b(215, 105, 245),
        Vec3b(85, 235, 235),
        Vec3b(245, 145, 115),
        Vec3b(120, 255, 45)
    };
    return palette[rand() % 8];
}

vector<RandomBand> make_random_bands(const RunSpec& run) {
    vector<RandomBand> bands;
    int band_count = 3 + (rand() % 2);
    double drift = run.drift;
    double intensity = run.intensity;

    for (int i = 0; i < band_count; ++i) {
        double center = 25.0 + static_cast<double>(rand() % 390);
        double width = 28.0 + static_cast<double>(rand() % 70);
        double rms_low = clamp_double(center - width * (0.45 + 0.15 * i), 0.0, 441.0);
        double rms_high = clamp_double(center + width * (0.55 + 0.10 * i), 0.0, 441.0);

        RandomBand band;
        band.rms_low = min(rms_low, rms_high);
        band.rms_high = max(rms_low, rms_high);
        band.target_color = random_awkward_color();
        band.mix_scale = clamp_double((0.30 + 0.12 * i) * intensity, 0.10, 0.95);
        band.passes = 1 + (rand() % 2);
        band.mix_pattern = random_pattern_name();
        band.shift_pattern_x = random_pattern_name();
        band.shift_pattern_y = random_pattern_name();
        band.shift_radius = max(2, static_cast<int>(round((2.0 + i) * drift)));
        bands.push_back(band);
    }

    return bands;
}

void apply_blend_burst(Mat& image, const vector<string>& recipes) {
    for (const string& recipe : recipes) {
        apply_blend_recipe(image, 1, recipe);
    }
}

string pick_random_shading(const vector<string>& pool, const string& avoid = "") {
    vector<string> candidates;
    for (const string& item : pool) {
        if (item != avoid) {
            candidates.push_back(item);
        }
    }
    if (candidates.empty()) {
        return pool.front();
    }
    return candidates[rand() % candidates.size()];
}

void apply_shading_pass(Mat& image, int width, int height, const string& shading, const RunSpec& run, int phase_seed){
    RunProfile profile = make_run_profile(shading);
    int num_it = scaled_passes(profile.base_passes, run.intensity);
    double intensity = run.intensity;
    double drift = run.drift;
    int drift_step = max(1, static_cast<int>(round(drift)));
    vector<RandomBand> bands = make_random_bands(run);
    for (int i = 0; i < num_it; ++i) {
        int x_phase = phase_seed + i * drift_step;
        int y_phase = phase_seed + static_cast<int>(round(i * drift));

        if (shading == "moody") {
            for (size_t b = 0; b < bands.size(); ++b) {
                const RandomBand& band = bands[b];
                shift_colorize(image, width, height, band.passes, band.rms_low, band.rms_high, band.target_color,
                    clamp_double(band.mix_scale * 1.05, 0.0, 1.0), band.mix_pattern, band.shift_pattern_x, band.shift_pattern_y,
                    band.shift_radius + 1, x_phase, y_phase * (b + 1));
            }
            apply_blend_burst(image, {"north_pull", "halo", "cross_stitch", "north_pull", "halo"});
        } else if (shading == "soft") {
            for (size_t b = 0; b < bands.size(); ++b) {
                const RandomBand& band = bands[b];
                shift_colorize(image, width, height, 1, band.rms_low, band.rms_high, band.target_color,
                    clamp_double(band.mix_scale * 0.70, 0.0, 1.0), band.mix_pattern, band.shift_pattern_y, band.shift_pattern_x,
                    max(1, band.shift_radius - 1), x_phase, y_phase + static_cast<int>(b));
            }
            apply_blend_burst(image, {"cross_stitch", "halo", "cross_stitch", "halo"});
        } else if (shading == "electric") {
            for (size_t b = 0; b < bands.size(); ++b) {
                const RandomBand& band = bands[b];
                shift_colorize(image, width, height, band.passes + 1, band.rms_low, band.rms_high, band.target_color,
                    clamp_double(band.mix_scale * 1.20, 0.0, 1.0), "scan_bars", band.shift_pattern_x, band.shift_pattern_y,
                    band.shift_radius + 2, x_phase * (b + 1), y_phase);
            }
            apply_blend_burst(image, {"scanline", "stutter_step", "cross_stitch", "scanline", "stutter_step", "cross_stitch"});
        } else if (shading == "wash") {
            for (size_t b = 0; b < bands.size(); ++b) {
                const RandomBand& band = bands[b];
                shift_colorize(image, width, height, band.passes, band.rms_low, band.rms_high, band.target_color,
                    clamp_double(band.mix_scale * 0.85, 0.0, 1.0), "corner_bloom", band.shift_pattern_y, band.shift_pattern_x,
                    band.shift_radius, x_phase, y_phase + static_cast<int>(b * 2));
            }
            apply_blend_burst(image, {"halo", "cross_stitch", "halo", "cross_stitch", "north_pull"});
        } else {
            for (size_t b = 0; b < bands.size(); ++b) {
                const RandomBand& band = bands[b];
                shift_colorize(image, width, height, band.passes, band.rms_low, band.rms_high, band.target_color,
                    band.mix_scale, band.mix_pattern, band.shift_pattern_x, band.shift_pattern_y,
                    band.shift_radius, x_phase + static_cast<int>(b), y_phase * (b + 1));
            }
            if (i % 2 == 0) {
                apply_blend_burst(image, {"cross_stitch", "north_pull", "halo", "cross_stitch", "scanline"});
            } else {
                apply_blend_burst(image, {"scanline", "stutter_step", "cross_stitch", "halo", "stutter_step"});
            }
        }

    }
}

vector<string> random_shading_pair() {
    vector<string> pool = available_shadings();
    string first = pick_random_shading(pool);
    string second = pick_random_shading(pool, first);
    return {first, second};
}

bool ends_with(const string& value, const string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

string strip_extension(const string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos == string::npos) {
        return filename;
    }
    return filename.substr(0, pos);
}

vector<string> list_source_pngs(const string& directory) {
    vector<string> files;
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        throw runtime_error("Could not open directory: " + directory);
    }

    dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        string name = entry->d_name;
        bool generated = name.find("_mix_") != string::npos;
        if (ends_with(name, ".png") && !generated) {
            files.push_back(name);
        }
    }
    closedir(dir);
    sort(files.begin(), files.end());
    return files;
}

void ensure_directory(const string& directory) {
    struct stat info;
    if (stat(directory.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        return;
    }
    if (mkdir(directory.c_str(), 0755) != 0) {
        throw runtime_error("Could not create directory: " + directory);
    }
}

int main() {
    string workspace = "output";
    string runs_path = "runs.txt";
    srand(time(0));
    auto batch_start = high_resolution_clock::now();

    ensure_directory(workspace);

    vector<RunSpec> runs = read_runs_file(runs_path);
    vector<string> source_files = list_source_pngs(workspace);
    if (source_files.empty()) {
        cerr << term::red << term::bold << "No source PNGs found" << term::reset
             << term::dim << "  put JPGs in ./input and run make prepare" << term::reset << endl;
        return 1;
    }

    print_run_plan(runs);
    print_source_plan(source_files, workspace);

    for (const string& filename : source_files) {
        string input_path = workspace + "/" + filename;
        Mat source = imread(input_path, IMREAD_COLOR);
        if (source.empty()) {
            cerr << term::red << "Could not load image: " << term::reset << input_path << endl;
            continue;
        }

        cout << term::bold << term::cyan << "Image" << term::reset << "  " << input_path << endl;
        cout << term::dim << "  size: " << source.cols << " x " << source.rows << term::reset << endl;

        for (const RunSpec& run : runs) {
            auto run_start = high_resolution_clock::now();
            Mat image = source.clone();
            vector<string> shading_pair = random_shading_pair();
            string first_shading = shading_pair[0];
            string second_shading = shading_pair[1];
            ostringstream suffix;
            suffix << "_" << run.label
                   << "_mix_" << first_shading << "_" << second_shading
                   << "_i" << static_cast<int>(round(run.intensity * 100))
                   << "_d" << static_cast<int>(round(run.drift * 100));
            string output_path = workspace + "/" + strip_extension(filename) + suffix.str() + ".png";

            cout << term::yellow << "  -> " << term::reset
                 << term::bold << run.label << term::reset
                 << term::dim << "  random_bands=3-4"
                 << "  intensity=" << format_double(run.intensity)
                 << "  drift=" << format_double(run.drift)
                 << "  shadings=" << first_shading << " -> " << second_shading
                 << term::reset << endl;

            apply_shading_pass(image, image.cols, image.rows, first_shading, run, rand() % 19);
            apply_shading_pass(image, image.cols, image.rows, second_shading, run, rand() % 31 + 7);
            imwrite(output_path, image);
            long long run_us = duration_cast<microseconds>(high_resolution_clock::now() - run_start).count();
            cout << term::green << "     wrote" << term::reset << "  " << output_path
                 << term::dim << "  in " << format_seconds(run_us) << "s" << term::reset << endl;
        }
        print_divider();
    }

    long long batch_us = duration_cast<microseconds>(high_resolution_clock::now() - batch_start).count();
    cout << term::bold << term::green << "Batch Complete" << term::reset
         << term::dim << "  processed " << source_files.size() << " source image(s), "
         << runs.size() << " run profile(s), total time " << format_seconds(batch_us) << "s"
         << term::reset << endl;

    return 0;
}

/*
    Linux setup:
    sudo apt install g++ make pkgconf libopencv-dev ffmpeg

    Workflow:
    1. Put raw .jpg/.jpeg files in ./input
    2. Run `make prepare` to convert them to .png files in ./output
    3. Edit ./runs.txt with `label intensity drift`
    4. Run `make run` to process every source .png in ./output
    5. Each run spec is shaded twice using a random pair of shading profiles

    The old macOS opencv2.framework file can be ignored on Linux.
*/
