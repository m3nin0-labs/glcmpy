#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <vector>

#include "glcm_core.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Thin wrapper so every measure shares one signature at binding layer
template <typename Metric>
glcmpy::Image run(const Eigen::Ref<const glcmpy::Image>& img,
                  const std::vector<double>& angles,
                  int n_grey,
                  int window_size) {
    return glcmpy::glcm_apply<Metric>(img, angles, n_grey, window_size);
}

}  // namespace

NB_MODULE(_glcm, m) {
    m.doc() = "GLCM texture measures (Core module)";

    constexpr const char* kDoc =
        "Compute a GLCM texture measure over a 2-D grey-level image.\n\n"
        "Parameters\n----------\n"
        "image  : 2-D float array, quantized to integers in [0, n_grey)\n"
        "angles : sequence of directions in radians\n"
        "n_grey : number of grey levels\n"
        "window_size : odd sliding-window size\n";

    m.def("contrast", &run<glcmpy::Contrast>, "image"_a, "angles"_a,
          "n_grey"_a, "window_size"_a, kDoc);
    m.def("dissimilarity", &run<glcmpy::Dissimilarity>, "image"_a, "angles"_a,
          "n_grey"_a, "window_size"_a, kDoc);
    m.def("homogeneity", &run<glcmpy::Homogeneity>, "image"_a, "angles"_a,
          "n_grey"_a, "window_size"_a, kDoc);
    m.def("energy", &run<glcmpy::Energy>, "image"_a, "angles"_a, "n_grey"_a,
          "window_size"_a, kDoc);
    m.def("asm", &run<glcmpy::Asm>, "image"_a, "angles"_a, "n_grey"_a,
          "window_size"_a, kDoc);
    m.def("mean", &run<glcmpy::Mean>, "image"_a, "angles"_a, "n_grey"_a,
          "window_size"_a, kDoc);
    m.def("variance", &run<glcmpy::Variance>, "image"_a, "angles"_a,
          "n_grey"_a, "window_size"_a, kDoc);
    m.def("std", &run<glcmpy::Std>, "image"_a, "angles"_a, "n_grey"_a,
          "window_size"_a, kDoc);
    m.def("correlation", &run<glcmpy::Correlation>, "image"_a, "angles"_a,
          "n_grey"_a, "window_size"_a, kDoc);
}
