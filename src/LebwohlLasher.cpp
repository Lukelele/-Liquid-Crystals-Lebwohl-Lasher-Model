#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;



class EigenSolver {
private:
    static constexpr double EPSILON = 1e-10;
    static constexpr int MAX_ITERATIONS = 100;
    
    int n;
    std::vector<std::vector<double>> matrix;
    std::vector<std::vector<double>> eigenvectors;
    std::vector<double> eigenvalues;

    void initializeEigenvectors() {
        eigenvectors = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
        for(int i = 0; i < n; i++) {
            eigenvectors[i][i] = 1.0;
        }
    }

    void rotate(int p, int q) {
        double diff = matrix[q][q] - matrix[p][p];
        double theta = 0.5 * atan2(2 * matrix[p][q], diff);
        double c = cos(theta);
        double s = sin(theta);

        for(int i = 0; i < n; i++) {
            double temp1 = eigenvectors[i][p];
            double temp2 = eigenvectors[i][q];
            eigenvectors[i][p] = c * temp1 - s * temp2;
            eigenvectors[i][q] = s * temp1 + c * temp2;
        }

        for(int i = 0; i < n; i++) {
            if(i != p && i != q) {
                double temp1 = matrix[p][i];
                double temp2 = matrix[q][i];
                matrix[i][p] = matrix[p][i] = c * temp1 - s * temp2;
                matrix[i][q] = matrix[q][i] = s * temp1 + c * temp2;
            }
        }

        double temp = matrix[p][p];
        matrix[p][p] = c * c * temp + s * s * matrix[q][q] - 2 * c * s * matrix[p][q];
        matrix[q][q] = s * s * temp + c * c * matrix[q][q] + 2 * c * s * matrix[p][q];
        matrix[p][q] = matrix[q][p] = 0.0;
    }

public:
    EigenSolver(const std::vector<std::vector<double>>& input_matrix) {
        n = input_matrix.size();
        matrix = input_matrix;
        eigenvalues.resize(n);
        initializeEigenvectors();
    }

    void compute() {
        for(int iter = 0; iter < MAX_ITERATIONS; iter++) {
            double max_element = 0.0;
            int p = 0, q = 0;

            for(int i = 0; i < n-1; i++) {
                for(int j = i+1; j < n; j++) {
                    if(std::abs(matrix[i][j]) > max_element) {
                        max_element = std::abs(matrix[i][j]);
                        p = i;
                        q = j;
                    }
                }
            }

            if(max_element < EPSILON) break;
            rotate(p, q);
        }

        for(int i = 0; i < n; i++) {
            eigenvalues[i] = matrix[i][i];
        }
    }

    std::vector<double> getEigenvalues() const { return eigenvalues; }
    std::vector<std::vector<double>> getEigenvectors() const { return eigenvectors; }
};



vector<vector<double> > initdat(int nmax){

    std::vector<std::vector<double> > arr(nmax, std::vector<double>(nmax));
    for (int i = 0; i < nmax; i++) {
        for (int j = 0; j < nmax; j++) {
            arr[i][j] = 2.0 * M_PI * (double)rand() / RAND_MAX;
        }
    }

    return arr;
}


double one_energy(vector<vector<double> > &arr, int ix, int iy, int nmax) {
    double en = 0.0;
    double ang = 0.0;
    
    // Correct way to handle periodic boundaries
    int ixp = (ix + 1) % nmax;
    int ixm = (ix - 1 + nmax) % nmax;  // Add nmax before modulo
    int iyp = (iy + 1) % nmax;
    int iym = (iy - 1 + nmax) % nmax;  // Add nmax before modulo

    // Rest of the function remains the same
    ang = arr[ix][iy]-arr[ixp][iy];
    en += 0.5 * (1.0 - 3.0 * std::cos(ang) * std::cos(ang));
    ang = arr[ix][iy]-arr[ixm][iy];
    en += 0.5 * (1.0 - 3.0 * std::cos(ang) * std::cos(ang));
    ang = arr[ix][iy]-arr[ix][iyp];
    en += 0.5 * (1.0 - 3.0 * std::cos(ang) * std::cos(ang));
    ang = arr[ix][iy]-arr[ix][iym];
    en += 0.5 * (1.0 - 3.0 * std::cos(ang) * std::cos(ang));

    return en;
}



double all_energy(vector<vector<double> > &arr, int nmax) {

    double enall = 0.0;

    for (int i = 0; i < nmax; i++) {
        for (int j = 0; j < nmax; j++) {
            enall += one_energy(arr,i,j,nmax);
        }
    }

    return enall;
}


double MC_step(vector<vector<double> > &arr, double Ts, int nmax) {

    double scale = 0.1 + Ts;
    int accept = 0;

    // Create 2D arrays using vectors
    std::vector<std::vector<int>> xran(nmax, std::vector<int>(nmax));
    std::vector<std::vector<int>> yran(nmax, std::vector<int>(nmax));
    std::vector<std::vector<double>> aran(nmax, std::vector<double>(nmax));

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> uniform_dist(0, nmax - 1);
    std::normal_distribution<> normal_dist(0.0, scale);

    for (int i = 0; i < nmax; ++i) {
        for (int j = 0; j < nmax; ++j) {
            xran[i][j] = uniform_dist(gen);
            yran[i][j] = uniform_dist(gen);
            aran[i][j] = normal_dist(gen);
        }
    }

    for (int i = 0; i < nmax; i++) {
        for (int j = 0; j < nmax; j++) {
            int ix = xran[i][j];
            int iy = yran[i][j];
            double ang = aran[i][j];
            double en0 = one_energy(arr,ix,iy,nmax);
            arr[ix][iy] += ang;
            double en1 = one_energy(arr,ix,iy,nmax);
            if (en1<=en0) {
                accept += 1;
            }
            else {
                double boltz = exp( -(en1 - en0) / Ts );

                if (boltz >= (double)rand() / RAND_MAX) {
                    accept += 1;
                }
                else {
                    arr[ix][iy] -= ang;
                }
            }
        }
    }

    return accept/(nmax*nmax);
}



double get_order(vector<vector<double> > &arr, int nmax) {

    std::vector<std::vector<double>> Qab(3, std::vector<double>(3, 0.0));

    std::vector<std::vector<double>> delta(3, std::vector<double>(3, 0.0));
    for (int i = 0; i < 3; ++i) {
        delta[i][i] = 1.0;
    }

    // Corrected initialization sizes
    std::vector<double> cosArr(nmax * nmax);
    std::vector<double> sinArr(nmax * nmax);
    std::vector<double> zerosArr(nmax * nmax, 0.0);

    // Fill cosArr and sinArr with cosine and sine values of arr
    for (int i = 0; i < nmax; ++i) {
        for (int j = 0; j < nmax; ++j) {
            cosArr[i * nmax + j] = cos(arr[i][j]);
            sinArr[i * nmax + j] = sin(arr[i][j]);
        }
    }

    // Create a 3D vector to hold the final result
    std::vector<std::vector<std::vector<double>>> lab(3, std::vector<std::vector<double>>(nmax, std::vector<double>(nmax)));

    // Reshape and fill the 3D vector
    for (int i = 0; i < nmax; ++i) {
        for (int j = 0; j < nmax; ++j) {
            lab[0][i][j] = cosArr[i * nmax + j];
            lab[1][i][j] = sinArr[i * nmax + j];
            lab[2][i][j] = zerosArr[i * nmax + j];
        }
    }

    // Calculate Qab matrix
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < nmax; ++i) {
                for (int j = 0; j < nmax; ++j) {
                    Qab[a][b] += 3 * lab[a][i][j] * lab[b][i][j] - delta[a][b];
                }
            }
        }
    }

    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            Qab[a][b] = Qab[a][b] / (nmax * nmax);
        }
    }

    EigenSolver eigenSolver(Qab);
    eigenSolver.compute();
    std::vector<double> eigenvalues = eigenSolver.getEigenvalues();
    
    int maxEigenvalue = *max_element(eigenvalues.begin(), eigenvalues.end());

    return maxEigenvalue;
}



int main() {
    int nsteps = 50;
    int nmax = 400;
    float temp = 0.5;
    int pflag = 0;

    vector<vector<double> > lattice = initdat(nmax);

    vector<double> energy;
    vector<double> ratio;
    vector<double> order;


    energy.push_back(all_energy(lattice,nmax));
    ratio.push_back(0.5);
    order.push_back(get_order(lattice,nmax));


    clock_t initial = clock();
    for (int it = 1; it < nsteps+1; it++) {
        ratio.push_back(MC_step(lattice,temp,nmax));
        energy.push_back(all_energy(lattice,nmax));
        order.push_back(get_order(lattice,nmax));
    }
    clock_t final = clock();

    double runtime = (double)(final - initial) / CLOCKS_PER_SEC;

    cout << "Size: " << nmax << ", Steps: " << nsteps << ", T*: " << temp << ", Order: " << order[nsteps-1] << ", Time: " << runtime << " s" << endl;
    cin.get();

    return 0;
}