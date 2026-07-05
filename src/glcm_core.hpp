// glcmpy - GLCM texture measures (core module)
//
// This is a pure C++/Eigen port of the algorithm in the `sits` R package
// (src/glcm_fns.cpp), which in turn was inspired by the `glcm` and
// `GLCMTextures` R packages (GPL >= 3) and scikit-image (BSD-3-Clause).
//
// The original used Armadillo + Rcpp. Here we depend only on Eigen and the
// C++ standard library so the code can be bound to Python with nanobind and
// built without R, BLAS, or LAPACK.

#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <vector>

namespace glcmpy {

// Row-major float matrix == numpy default C-contiguous layout, so the
// numpy <-> Eigen boundary is zero-copy on input
using Image = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>;

// A single observed co-occurrence cell: grey levels (i, j) with count
// For a `w x w` window there are at most `w*w` such cells, which keeps every
// per-pixel reduction tiny regardless of how large `n_grey` is. This mirrors
// the sparse co-occurrence matrix used in the R implementation
struct CoocCell {
    int i;
    int j;
    double count;
};

// Symmetric co-occurrence list for one `window + angle` plus its total count
struct Cooccurrence {
    std::vector<CoocCell> cells;
    double total = 0.0;
};

// Accumulate one `(i, j)` co-occurrence into the list, merging duplicates
inline void add_pair(Cooccurrence& g, int i, int j);

// Metric functors
// Each takes the (normalized) co-occurrence list and returns one scalar. They
// reproduce the equations in `glcm_fns.cpp:115-195`. 
//  - `p` is the probability of a cell (`count / total`);
//  - `i`, `j` are the grey levels of that cell
struct Contrast {
    static double reduce(const Cooccurrence& g) {
        double res = 0.0;
        
        for (const auto& c : g.cells) {
            const double d = static_cast<double>(c.i - c.j);

            res += (c.count / g.total) * d * d;
        }

        return res;
    }
};

struct Dissimilarity {
    static double reduce(const Cooccurrence& g) {
        double res = 0.0;
        
        for (const auto& c : g.cells) {
            res += (c.count / g.total) * std::abs(c.i - c.j);
        }

        return res;
    }
};

struct Homogeneity {
    static double reduce(const Cooccurrence& g) {
        double res = 0.0;

        for (const auto& c : g.cells) {
            const double d = static_cast<double>(c.i - c.j);

            res += (c.count / g.total) / (1.0 + d * d);
        }

        return res;
    }
};

struct Asm {
    static double reduce(const Cooccurrence& g) {
        double res = 0.0;
        
        for (const auto& c : g.cells) {
            const double p = c.count / g.total;

            res += p * p;
        }
        
        return res;
    }
};

struct Energy {
    static double reduce(const Cooccurrence& g) {
        return std::sqrt(Asm::reduce(g));
    }
};

struct Mean {
    static double reduce(const Cooccurrence& g) {
        double res = 0.0;
        
        for (const auto& c : g.cells) { 
            res += (c.count / g.total) * c.i;
        }

        return res;
    }
};

struct Variance {
    static double reduce(const Cooccurrence& g) {
        const double mean = Mean::reduce(g);
        double res = 0.0;

        for (const auto& c : g.cells) {
            const double d = c.i - mean;

            res += (c.count / g.total) * d * d;
        }

        return res;
    }
};

struct Std {
    static double reduce(const Cooccurrence& g) {
        return std::sqrt(Variance::reduce(g));
    }
};

struct Correlation {
    static double reduce(const Cooccurrence& g) {
        const double mean = Mean::reduce(g);
        const double var = Variance::reduce(g);

        // Special-case near-zero variance, following scikit-image (and the
        // sits implementation, `glcm_fns.cpp:189-193`).
        if (var < 1e-15) {
            return 1.0;
        }
            
        double res = 0.0;
        for (const auto& c : g.cells) {
            res += (c.count / g.total) * ((c.i - mean) * (c.j - mean) / var);
        }

        return res;
    }
};

// Mirrored edge indices
// Reproduces `locus_neigh2` (`glcm_fns.cpp:11-22`): extends an axis of length
// `size` by `leg` on each side, reflecting indices at the borders.
inline std::vector<int> locus_neigh(int size, int leg) {
    std::vector<int> res(static_cast<std::size_t>(size + 2 * leg));

    // iterate over the result vector
    for (int i = 0; i < static_cast<int>(res.size()); ++i) {

        // case: index is less than the leg
        if (i < leg) {
            res[i] = leg - i - 1;
        }
        
        // case: index is less than the size + leg
        else if (i < size + leg) {
            res[i] = i - leg;
        }

        // case: index is greater than the size + leg
        else {
            res[i] = 2 * size + leg - i - 1;
        }
    }

    // return!
    return res;
}

// Core engine
// Slides a `window_size x window_size` window over `img`, builds a symmetric
// co-occurrence list per pixel/angle, reduces it with Metric, and averages
// over angles. Returns a new same-shape matrix (never mutates the input).
template <typename Metric>
Image glcm_apply(const Eigen::Ref<const Image>& img,
                 const std::vector<double>& angles,
                 int n_grey,
                 int window_size) {
    // get the number of rows and columns of the image
    const int nrows = static_cast<int>(img.rows());
    const int ncols = static_cast<int>(img.cols());

    // get the leg of the window
    const int leg = window_size / 2;

    // get the number of angles
    const int n_ang = static_cast<int>(angles.size());

    // get the mirrored edge indices for the rows
    const std::vector<int> loci = locus_neigh(nrows, leg);

    // get the mirrored edge indices for the columns
    const std::vector<int> locj = locus_neigh(ncols, leg);

    // create the output image
    Image out(nrows, ncols);

    // check if OpenMP is available
#ifdef GLCMPY_HAVE_OPENMP
#pragma omp parallel
#endif
    {
        // per-thread scratch reused across pixels
        Cooccurrence cooc;
        cooc.cells.reserve(static_cast<std::size_t>(window_size * window_size));

        // create the neighbourhood vector
        std::vector<int> neigh(
            static_cast<std::size_t>(window_size * window_size));

#ifdef GLCMPY_HAVE_OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        // iterate over the rows
        for (int i = 0; i < nrows; ++i) {

            // iterate over the columns
            for (int j = 0; j < ncols; ++j) {

                // initialize the accumulator
                double acc = 0.0;

                // iterate over the angles
                for (int a = 0; a < n_ang; ++a) {
                    // get the angle
                    const double ang = angles[static_cast<std::size_t>(a)];

                    // gather the (mirrored) neighbourhood as integer grey
                    // levels (`glcm_fns.cpp:70-75`)
                    for (int wi = 0; wi < window_size; ++wi) {
                        // get the row index
                        const int ri = loci[static_cast<std::size_t>(wi + i)];

                        // iterate over the columns
                        for (int wj = 0; wj < window_size; ++wj) {
                            // get the column index
                            const int rj =
                                locj[static_cast<std::size_t>(wj + j)];

                            // set the neighbourhood value
                            neigh[static_cast<std::size_t>(wi * window_size +
                                                           wj)] =
                                static_cast<int>(img(ri, rj));
                        }
                    }

                    // get the row offset
                    const int off_row =
                        static_cast<int>(std::lround(std::sin(ang)));

                    // get the column offset
                    const int off_col =
                        static_cast<int>(std::lround(std::cos(ang)));

                    // get the start row
                    const int start_row = std::max(0, -off_row);

                    // get the end row
                    const int end_row =
                        std::min(window_size, window_size - off_row);

                    // get the start column
                    const int start_col = std::max(0, -off_col);

                    // get the end column
                    const int end_col =
                        std::min(window_size, window_size - off_col);

                    // clear the co-occurrence list
                    cooc.cells.clear();
                    cooc.total = 0.0;

                    // iterate over the rows
                    for (int r = start_row; r < end_row; ++r) {
                        for (int c = start_col; c < end_col; ++c) {
                            // get the value at the current position
                            const int vi =
                                neigh[static_cast<std::size_t>(r * window_size +
                                                               c)];

                            // get the value at the current position
                            const int vj = neigh[static_cast<std::size_t>(
                                (r + off_row) * window_size + (c + off_col))];

                            // check if the values are within the range
                            if (vi < n_grey && vj < n_grey) {
                                // symmetric matrix: count both directions
                                // (`glcm_fns.cpp:92-93`)

                                add_pair(cooc, vi, vj);
                                add_pair(cooc, vj, vi);
                            }
                        }
                    }

                    // check if the total is greater than 0
                    if (cooc.total > 0.0) {
                        // reduce the co-occurrence list
                        acc += Metric::reduce(cooc);
                    }
                }

                // set the value in the output image
                out(i, j) = (n_ang > 0) ? acc / n_ang : 0.0;
            }
        }
    }

    // end
    return out;
}

// add a pair to the co-occurrence list
inline void add_pair(Cooccurrence& g, int i, int j) {
    // iterate over the cells
    for (auto& c : g.cells) {
        // check if the cell is already in the list
        if (c.i == i && c.j == j) {
            // increment the count
            c.count += 1.0;
            g.total += 1.0;

            return;
        }
    }

    // add the cell to the list
    g.cells.push_back(CoocCell{i, j, 1.0});
    g.total += 1.0;
}

}  // namespace glcmpy
