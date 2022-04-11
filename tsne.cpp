#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void usage(const char *progname) {
    printf("Program Options:\n");
    printf("  -D  --dataset   <FILENAME>   Input file containing the raw feature data\n");
    printf("  -d  --nn_dists  <FILENAME>   Input file with L2 distances between nearest neighbors\n");
    printf("  -i  --nn_index  <FILENAME>   Input file identifying indices of nearest neighbors\n");
    printf("  -k  --k         <INT>        Number of nearest neighbors per point\n");
    printf("  -h  --help                   This message\n");
}

template <class T>
std::vector<T> load_data(std::string filename, T (*convert)(const std::string&)) {
    std::vector<T> vec;
    std::ifstream in_file(filename.c_str());
    if (in_file.is_open()) {
        std::string line, token;
        while (std::getline(in_file, line)) {
            std::stringstream ss(line);
            while (std::getline(ss, token, ',')) {
                vec.push_back(convert(token));
            }
        }
        in_file.close();
    }
    else {
        printf("Could not open file %s\n", filename.c_str());
        exit(1);
    }
    return vec;
}

int stoi(const std::string& s) {
    return std::stoi(s);
}

float stof(const std::string& s) {
    return std::stof(s);
}

int main(int argc, char **argv) {

    // Parse commandline options
    int opt;
    static struct option long_options[] = {
        {"dataset", 1, 0, 'D'},
        {"nn_dists", 1, 0, 'd'},
        {"nn_index", 1, 0, 'i'},
        {"k", 1, 0, 'k'},
        {"help", 0, 0, 'h'},
        {0, 0, 0, 0}
    };

    int k;
    std::string dataset_fname;
    std::string dists_fname;
    std::string index_fname;

    while ((opt = getopt_long(argc, argv, "D:d:i:k:h", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'D':
            dataset_fname = optarg;
            break;
        case 'd':
            dists_fname = optarg;
            break;
        case 'i':
            index_fname = optarg;
            break;
        case 'k':
            k = atoi(optarg);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    std::vector<int> nn_index = load_data(index_fname, &stoi);
    std::vector<float> nn_dists = load_data(dists_fname, &stof);
    std::vector<float> dataset = load_data(dataset_fname, &stof);

    return 0;
}
