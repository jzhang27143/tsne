#include <cuda.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "kernels/gradients.h"
#include "kernels/perplexity_search.h"
#include "kernels/utils.h"

void usage(const char *progname) {
    printf("Program Options:\n");
    printf("  -D  --dataset     <FILENAME>   Input file containing the raw feature data\n");
    printf("  -d  --nn_dists    <FILENAME>   Input file with L2 distances between nearest neighbors\n");
    printf("  -i  --nn_index    <FILENAME>   Input file identifying indices of nearest neighbors\n");
    printf("  -k  --k           <INT>        Number of nearest neighbors per point\n");
    printf("  -p  --perplexity  <FLOAT>      Perplexity target for variance initialization\n");
    printf("  -e  --epsilon     <FLOAT>      Convergence threshold for perplexity search\n");
    printf("  -T  --num_iters   <INT>        Number of gradient descent iterations\n");
    printf("  -h  --help                     This message\n");
}

template <class T>
thrust::device_vector<T> load_data(std::string filename, T (*convert)(const std::string&)) {
    // May need device_malloc for MNIST
    thrust::device_vector<T> vec;
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

void dump_final_embeds(std::string dataset_fname, int num_points,
                       thrust::host_vector<float> &embed_x,
                       thrust::host_vector<float> &embed_y) {
    
    std::string input_fname(dataset_fname);

    input_fname = input_fname.substr(5, input_fname.length() - 5 - 4);
    std::string output_fname = "./output_" + input_fname + ".txt";
    std::cout << output_fname << "\n";
    FILE *output = fopen(const_cast<char*>(output_fname.c_str()), "w");
    if (!output) {
        std::cout << "Unable to create file\n";
    }
    
    for (int i = 0; i < num_points; i++) {
        fprintf(output, "%.6f %.6f\n", embed_x[i], embed_y[i]);
    }

    fclose(output);
}

int main(int argc, char **argv) {

    // Parse commandline options
    int opt;
    static struct option long_options[] = {
        {"dataset", 1, 0, 'D'},
        {"nn_dists", 1, 0, 'd'},
        {"nn_index", 1, 0, 'i'},
        {"k", 1, 0, 'k'},
        {"perplexity", 1, 0, 'p'},
        {"theta", 1, 0, 't'},
        {"epsilon", 1, 0, 'e'},
        {"num_iters", 1, 0, 'T'},
        {"help", 0, 0, 'h'},
        {0, 0, 0, 0}
    };

    int k;
    int num_iters = 1000;
    float perplexity_target = 30.f;
    float epsilon = 1e-4;
    float theta = 0.5;
    std::string dataset_fname;
    std::string dists_fname;
    std::string index_fname;

    while ((opt = getopt_long(argc, argv, "D:d:i:k:p:t:e:T:h", long_options, NULL)) != EOF) {
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
        case 'p':
            perplexity_target = atof(optarg);
            break;
        case 't':
            theta = atof(optarg);
            break;
        case 'e':
            epsilon = atof(optarg);
            break;
        case 'T':
            num_iters = atoi(optarg);
            break;
        case 'h':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    thrust::device_vector<int> nn_index = load_data(index_fname, &stoi);
    thrust::device_vector<float> nn_dists = load_data(dists_fname, &stof);
    thrust::device_vector<float> dataset = load_data(dataset_fname, &stof);
    int num_points = nn_index.size() / k;

    thrust::device_vector<float> pij(num_points * k);
    thrust::device_vector<float> pij_sym(num_points * k);
    search_perplexity(pij, nn_dists, perplexity_target, epsilon, num_points, k);
    symmetrize_matrix(pij_sym, pij, nn_index, num_points, k);
    
    // Initialize 2D points
    thrust::device_vector<float> embed_x(num_points);
    thrust::device_vector<float> embed_y(num_points);
    initialize_points(embed_x, embed_y, num_points);

    // Initialize attractive and repulsive forces
    thrust::device_vector<float> grad_attract_x(num_points);
    thrust::device_vector<float> grad_attract_y(num_points);
    thrust::device_vector<float> grad_repulsive_x(num_points);
    thrust::device_vector<float> grad_repulsive_y(num_points);

    // Initialize parameters for t-SNE
    thrust::device_vector<float> old_forces_x(num_points, 0.f);
    thrust::device_vector<float> old_forces_y(num_points, 0.f);
    thrust::device_vector<float> gains_x(num_points, 1.f);
    thrust::device_vector<float> gains_y(num_points, 1.f);
    
    float learning_rate = 200.f;
    float momentum = 0.5f;
    float exaggeration = 12.f;
    
    for (int t = 0; t < num_iters; t++) {

        std::cout << "Iteration: " << t << std::endl;
        compute_attractive_forces(pij_sym, embed_x, embed_y, nn_index,
                                  grad_attract_x, grad_attract_y, num_points, k);
        compute_repulsive_forces(embed_x, embed_y, grad_repulsive_x, grad_repulsive_y,
                                 num_points, theta);
        if (t > 250) {
            momentum = 0.8f;
            exaggeration = 1.f;
        }
        apply_forces(embed_x, embed_y, gains_x, gains_y, old_forces_x, old_forces_y,
                     grad_attract_x, grad_attract_y,
                     grad_repulsive_x, grad_repulsive_y,
                     learning_rate, momentum, exaggeration, num_points);
    }
    /*
    thrust::host_vector<float> host_x(9);
    thrust::host_vector<float> host_y(9);
    host_x[0] = 0.f;
    host_x[1] = 0.f;
    host_x[2] = 0.f;
    host_x[3] = 1.f;
    host_x[4] = 1.f;
    host_x[5] = 1.f;
    host_x[6] = 2.f;
    host_x[7] = 2.f;
    host_x[8] = 2.f;

    host_y[0] = 0.f;
    host_y[1] = 1.f;
    host_y[2] = 2.f;
    host_y[3] = 0.f;
    host_y[4] = 1.f;
    host_y[5] = 2.f;
    host_y[6] = 0.f;
    host_y[7] = 1.f;
    host_y[8] = 2.f;
 

    thrust::device_vector<float> device_x = host_x;
    thrust::device_vector<float> device_y = host_y;   
    thrust::device_vector<float> grad_repulsive_x(9, 0.f);
    thrust::device_vector<float> grad_repulsive_y(9, 0.f);  
    
    for (int i = 0; i < 9; i++) {
        std::cout << grad_repulsive_x[i] << " " << grad_repulsive_y[i] << std::endl;
    }  

    compute_repulsive_forces(device_x, device_y, grad_repulsive_x, grad_repulsive_y, 9, theta);
    thrust::host_vector<float> host_repulsive_x(9);
    thrust::host_vector<float> host_repulsive_y(9);

    thrust::copy(grad_repulsive_x.begin(), grad_repulsive_x.end(), host_repulsive_x.begin());
    thrust::copy(grad_repulsive_y.begin(), grad_repulsive_y.end(), host_repulsive_y.begin());

    for (int i = 0; i < 9; i++) {
        std::cout << host_repulsive_x[i] << " " << host_repulsive_y[i] << std::endl;
    }
    */
    thrust::host_vector<float> host_embed_x(num_points);
    thrust::host_vector<float> host_embed_y(num_points);
    thrust::copy(embed_x.begin(), embed_x.end(), host_embed_x.begin());
    thrust::copy(embed_y.begin(), embed_y.end(), host_embed_y.begin());

    dump_final_embeds(dataset_fname, num_points, host_embed_x, host_embed_y);         
    return 0;
}
