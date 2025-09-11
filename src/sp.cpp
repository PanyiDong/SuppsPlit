/*
 * File Name: sp.cpp
 * Author: Panyi Dong
 * GitHub: https://github.com/PanyiDong/
 * Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)
 * 
 * Project: src
 * Latest Version: <<projectversion>>
 * Relative Path: /sp.cpp
 * File Created: Thursday, 11th September 2025 2:44:42 pm
 * Author: Panyi Dong (panyid2@illinois.edu)
 * 
 * -----
 * Last Modified: Thursday, 11th September 2025 4:02:26 pm
 * Modified By: Panyi Dong (panyid2@illinois.edu)
 * 
 * -----
 * MIT License
 * 
 * Copyright (c) 2025 - 2025, Panyi Dong
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// sp.cpp
// Plain C++ / Armadillo version of original sp.cpp (Rcpp removed)
//
// Requires: Armadillo, optionally OpenMP
//
#include <vector>
#include <iostream>
#include <cmath>
#include <vector>
#include <float.h>
#include <random>
#include <chrono>

int parallel_threads = 1;
#ifdef _OPENMP
  #include <omp.h>
#endif

void printProgress(int percent)
{
    if(parallel_threads == 1)
        std::cout << "\rOptimizing <1 thread> [" << std::string(percent / 5, '+') << std::string(100 / 5 - percent / 5, ' ') << "] " << percent << "%";
    else
        std::cout << "\rOptimizing <" << parallel_threads << " threads> [" << std::string(percent / 5, '+') << std::string(100 / 5 - percent / 5, ' ') << "] " << percent << "%";
    std::cout.flush();
}

std::vector<std::vector<double>> sp_cpp(std::size_t des_num, int dim_num,
                 const std::vector<std::vector<double>>& ini,
                 const std::vector<std::vector<double>>& distsamp,
                 bool thin,
                 const std::vector<std::vector<double>>& bd,
                 std::size_t point_num,
                 int it_max,
                 double tol,
                 int num_proc,
                 double n0,
                 const std::vector<double>& wts,
                 bool rnd_flg)
{
    #ifdef _OPENMP
        omp_set_num_threads(num_proc);
        parallel_threads = num_proc;
    #endif

    int it_num = 0;
    bool cont = true;

    std::vector<double> curconst(des_num, 0.0);
    std::vector<double> runconst(des_num, 0.0);
    std::vector<double> runconst_up(des_num, 0.0);

    // store designs as matrix des (des_num x dim_num)
    std::vector<std::vector<double>> des = ini;
    std::vector<std::vector<double>> des_up = des;
    std::vector<std::vector<double>> prevdes = des;

    double nug = 0.0;
    int percent_complete = 0;

    // RNG
    std::default_random_engine generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> uddist(0, (int)distsamp.size() - 1);

    while(cont)
    {
        int percent = (100 * (it_num + 1)) / it_max;
        if(percent > percent_complete)
        {
            printProgress(percent);
            percent_complete = percent;
        }

    std::fill(curconst.begin(), curconst.end(), 0.0);
    // removed unused variable nanflg
        prevdes = des;

        // prepare rnd sample and weights
        std::vector<std::vector<double>> rnd(point_num, std::vector<double>(dim_num, 0.0));
        std::vector<double> rnd_wts(point_num, 0.0);
        for(std::size_t i = 0; i < point_num; ++i)
        {
            std::size_t ss;
            if(rnd_flg)
                ss = (std::size_t)uddist(generator);
            else
                ss = i % distsamp.size();

            for(int j = 0; j < dim_num; ++j)
                rnd[i][j] = distsamp[ss][j];

            rnd_wts[i] = wts[ss];
        }

        // Parallel over design points
        #pragma omp parallel for schedule(static)
        for(std::size_t m = 0; m < des_num; ++m)
        {
            std::vector<double> xprime(dim_num, 0.0);
            std::vector<double> tmpvec(dim_num, 0.0);

            // interactions with other design points
            for(std::size_t o = 0; o < des_num; ++o)
            {
                if(o == m) continue;
                double tmptol = 0.0;
                for(int n = 0; n < dim_num; ++n)
                {
                    tmpvec[n] = prevdes[m][n] - prevdes[o][n];
                    tmptol += tmpvec[n] * tmpvec[n];
                }
                tmptol = std::sqrt(tmptol);
                for(int n = 0; n < dim_num; ++n)
                    xprime[n] += tmpvec[n] / (tmptol + nug * DBL_MIN);
            }

            for(int n = 0; n < dim_num; ++n)
                xprime[n] = xprime[n] * ((double)point_num / (double)des_num);

            double local_curconst = 0.0;

            // interactions with sample points
            for(std::size_t o = 0; o < point_num; ++o)
            {
                double tmptol = 0.0;
                for(int n = 0; n < dim_num; ++n)
                {
                    double diff = rnd[o][n] - prevdes[m][n];
                    tmptol += diff * diff;
                }
                tmptol = std::sqrt(tmptol);
                double denom = tmptol + (nug * DBL_MIN);
                local_curconst += rnd_wts[o] / denom;
                for(int n = 0; n < dim_num; ++n)
                    xprime[n] += rnd_wts[o] * rnd[o][n] / denom;
            }

            double denom = (1.0 - (n0 / (it_num + n0))) * runconst[m] + (n0 / (it_num + n0)) * local_curconst;
            if(denom == 0.0)
            {
                // produce NaN markers and continue
                for(int n = 0; n < dim_num; ++n)
                {
                    des_up[m][n] = std::numeric_limits<double>::quiet_NaN();
                }
                runconst_up[m] = 0.0;
                continue;
            }

            for(int n = 0; n < dim_num; ++n)
                xprime[n] = ((1.0 - (n0 / (it_num + n0))) * runconst[m] * prevdes[m][n] + (n0 / (it_num + n0)) * xprime[n] ) / denom;

            // enforce bounds
            for(int n = 0; n < dim_num; ++n)
            {
                double lower = bd[n][0];
                double upper = bd[n][1];
                double val = xprime[n];
                if(val < lower) val = lower;
                if(val > upper) val = upper;
                des_up[m][n] = val;
                if(std::isnan(val))
                {
                    // mark nan
                }
            }

            runconst_up[m] = (1 - (n0 / (it_num + n0))) * runconst[m] + (n0 / (it_num + n0)) * local_curconst;
        } // end parallel for

        // check for NaN
        bool any_nan = false;
        for (const auto& row : des_up) {
            for (double val : row) {
                if (std::isnan(val)) {
                    any_nan = true;
                    break;
                }
            }
            if (any_nan) break;
        }

        if(any_nan)
        {
            nug += 1.0;
            std::fill(runconst.begin(), runconst.end(), 0.0);
            std::cout << "\nNumerical instabilities encountered; resetting optimization.\n";
        }
        else
        {
            des = des_up;
            runconst = runconst_up;
        }

        it_num++;
        double maxdiff = 0.0;
        for(std::size_t n = 0; n < des_num; ++n)
        {
            double rundiff = 0.0;
            for(int o = 0; o < dim_num; ++o)
                rundiff += std::pow(des[n][o] - prevdes[n][o], 2.0);
            if(rundiff > maxdiff) maxdiff = rundiff;
        }

        if((maxdiff < tol) && (!any_nan))
        {
            cont = false;
            std::cout << "\nTolerance level reached.";
        }

        if((it_num >= it_max) && (!any_nan))
        {
            cont = false;
        }
    } // end while

    std::cout << "\n";
    return des;
}
