#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Dense>


using namespace std;



vector<vector<double> > initdat(int nmax){

    std::vector<std::vector<double> > arr(nmax, std::vector<double>(nmax));
    for (int i = 0; i < nmax; i++) {
        for (int j = 0; j < nmax; j++) {
            arr[i][j] = 2.0 * M_PI * (double)rand() / RAND_MAX;
        }
    }

    return arr;
}


float one_energy(vector<vector<double> > arr,int ix,int iy,int nmax){

    float en = 0.0;
    float ang = 0.0;
    float ixp = (ix+1)%nmax;
    float ixm = (ix-1)%nmax;
    float iyp = (iy+1)%nmax;
    float iym = (iy-1)%nmax;

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


float all_energy(vector<vector<double> > arr, int nmax) {

    double enall = 0.0;

    for (int i = 0; i < nmax; i++) {
        for (int j = 0; j < nmax; j++) {
            enall += one_energy(arr,i,j,nmax);
        }
    }
    
    return enall;
}


double MC_step(vector<vector<double> > arr, double Ts, int nmax) {

    double scale=0.1+Ts;
    double accept = 0;

    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());

    // Integer random numbers in the range [0, nmax)
    std::uniform_int_distribution<> dis_int(0, nmax - 1);

    // Normal distribution with mean 0 and standard deviation 'scale'
    std::normal_distribution<> dis_normal(0.0, scale);

    vector<vector<int> > xran(nmax, vector<int>(nmax));
    vector<vector<int> > yran(nmax, vector<int>(nmax));
    vector<vector<double> > aran(nmax, vector<double>(nmax));

    for (int i = 0; i < nmax; ++i) {
        for (int j = 0; j < nmax; ++j) {
            xran[i][j] = dis_int(gen);
            yran[i][j] = dis_int(gen);
            aran[i][j] = dis_normal(gen);
        }
    }

    for (int i = 0; i < nmax; i++) {
        for (int j = 0; j < nmax; j++) {
            double ix = xran[i][j];
            double iy = yran[i][j];
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



// float get_order(vector<vector<double> > arr, int nmax) {

//     std::vector<std::vector<double>> Qab(3, std::vector<double>(3, 0.0));

//     std::vector<std::vector<double>> delta(3, std::vector<double>(3, 0.0));
//     for (int i = 0; i < 3; ++i) {
//         delta[i][i] = 1.0;
//     }

//     // Create vectors to hold cos, sin, and zeros
//     std::vector<double> cosArr(arr.size());
//     std::vector<double> sinArr(arr.size());
//     std::vector<double> zerosArr(arr.size(), 0.0);

//     // Fill cosArr and sinArr with cosine and sine values of arr
//     for (size_t i = 0; i < arr.size(); ++i) {
//         for (size_t j = 0; j < arr[i].size(); ++j) {
//             cosArr[i * arr.size() + j] = cos(arr[i][j]);
//             sinArr[i * arr.size() + j] = sin(arr[i][j]);
//         }
//     }

//     // Create a 3D vector to hold the final result
//     std::vector<std::vector<std::vector<double>>> lab(3, std::vector<std::vector<double>>(nmax, std::vector<double>(nmax)));

//     // Reshape and fill the 3D vector
//     for (int i = 0; i < nmax; ++i) {
//         for (int j = 0; j < nmax; ++j) {
//             lab[0][i][j] = cosArr[i * nmax + j];
//             lab[1][i][j] = sinArr[i * nmax + j];
//             lab[2][i][j] = zerosArr[i * nmax + j];
//         }
//     }

//     for (int a = 0; a < 3; ++a) {
//         for (int b = 0; b < 3; ++b) {
//             for (int i = 0; i < nmax; ++i) {
//                 for (int j = 0; j < nmax; ++j) {
//                     // Qab[a,b] += 3 * lab[a,i,j] * lab[b,i,j] - delta[a,b];
//                 }
//             }
//         }
//     }

//     Qab = Qab/(2*nmax*nmax);

//     Eigen::EigenSolver<Eigen::MatrixXd> solver(Qab);
    
//     // Get the eigenvalues
//     Eigen::VectorXd eigenvalues = solver.eigenvalues().real();

//     // Find and return the maximum eigenvalue
//     return eigenvalues.maxCoeff();
// }


int main() {
    int nsteps = 10;
    int nmax = 10;
    float temp = 1.0;
    int pflag = 0;

    vector<vector<double> > lattice = initdat(nmax);

    vector<double> energy = vector<double>(nsteps+1, 0);
    vector<double> ratio = vector<double>(nsteps+1, 0);
    vector<double> order = vector<double>(nsteps+1, 0);

    energy[0] = all_energy(lattice,nmax);
    ratio[0] = 0.5;
    // order[0] = get_order(lattice,nmax);

    clock_t initial = clock();
    for (int it = 1; it < nsteps+1; it++) {
        ratio[it] = MC_step(lattice,temp,nmax);
        energy[it] = all_energy(lattice,nmax);
        // order[it] = get_order(lattice,nmax);
    }
    clock_t final = clock();

    double runtime = (double)(final - initial) / CLOCKS_PER_SEC;
    cout << runtime << endl;
    

    // print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime));

    return 0;
}