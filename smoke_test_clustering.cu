// Smoke test for TopologyClusterer::fit() (Phase III stage VI).
//
// Builds 3 well-separated synthetic 4-feature blobs (mimicking the
// S_YM / max|gamma| / mean|gamma| / beta1 layout fit() expects), uploads
// them to device, runs DBSCAN through TopologyClusterer, and checks that
// it found roughly 3 clusters with low noise. This does NOT exercise the
// real feature pipeline (stage V aggregation still doesn't exist) -- it
// only proves fit()/ensure_label_buffer() work end-to-end against a real
// cuML install: no crashes, plausible label output, sane timing.
//
// Build (from engine/, after `cmake --build build --target holo_ai`):
//
//   nvcc -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda \
//       -I. -isystem $CUML_ROOT/include \
//       -isystem $CUML_ROOT/include/rapids/libcudacxx \
//       -isystem /usr/local/cuda/targets/x86_64-linux/include \
//       ../smoke_test_clustering.cu build/libholo_ai.a \
//       -L$CUML_ROOT/lib -lcuml++ -Wl,-rpath,$CUML_ROOT/lib \
//       -DHOLO_HAS_CUML=1 \
//       -o /tmp/smoke_test_clustering
//
//   /tmp/smoke_test_clustering
//
// If your CMakeLists.txt already defines HOLO_HAS_CUML via
// target_compile_definitions on holo_ai (it does, from cuml_FOUND), you
// still need to pass -DHOLO_HAS_CUML=1 here explicitly since this file
// is compiled standalone, outside that target.

#include "ai/cuml_clustering.cuh"
#include "math/cuda_utils.cuh"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    using holo::cuda::ClusteringConfig;
    using holo::cuda::TopologyClusterer;

    constexpr int kFeatures = 4;   // S_YM, max|gamma|, mean|gamma|, beta1
    constexpr int kPerBlob  = 150;
    constexpr int kBlobs    = 3;
    constexpr int kSamples  = kPerBlob * kBlobs;

    // Centers spaced far apart relative to the noise std below, so this
    // is an easy separation problem -- the point is "does the plumbing
    // work", not "how good is DBSCAN".
    const std::vector<std::array<float, kFeatures>> centers = {
        {0.0F, 0.0F, 0.0F, 0.0F},
        {5.0F, 5.0F, 5.0F, 5.0F},
        {-5.0F, 5.0F, -5.0F, 5.0F},
    };

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0F, 0.15F);

    std::vector<float> h_features(static_cast<std::size_t>(kSamples) * kFeatures);
    for (int b = 0; b < kBlobs; ++b) {
        for (int i = 0; i < kPerBlob; ++i) {
            const int row = b * kPerBlob + i;
            for (int f = 0; f < kFeatures; ++f) {
                h_features[static_cast<std::size_t>(row) * kFeatures + f] =
                    centers[static_cast<std::size_t>(b)][static_cast<std::size_t>(f)] + noise(rng);
            }
        }
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate failed\n");
        return 1;
    }

    float* d_features = nullptr;
    if (cudaMallocAsync(reinterpret_cast<void**>(&d_features),
                         h_features.size() * sizeof(float), stream) != cudaSuccess) {
        std::fprintf(stderr, "cudaMallocAsync failed\n");
        return 1;
    }
    if (cudaMemcpyAsync(d_features, h_features.data(),
                         h_features.size() * sizeof(float),
                         cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyAsync H2D failed\n");
        return 1;
    }
    cudaStreamSynchronize(stream);

    // eps chosen generously relative to the 0.15 noise std above -- this
    // is a smoke test config, not the production ClusteringConfig{} in
    // the header (eps=0.15F is tuned for real normalized signal data,
    // not this synthetic blob spacing).
    ClusteringConfig cfg;
    cfg.eps = 1.0F;
    cfg.min_samples = 5;

    // `clusterer` holds `stream` internally (its destructor calls
    // cudaFreeAsync(..., stream_)), so it MUST be destroyed before the
    // stream is. Giving it its own nested scope guarantees that: this
    // inner block ends -- running ~TopologyClusterer() -- while `stream`
    // is still alive, well before cudaStreamDestroy(stream) below. Do not
    // "flatten" this back into main()'s top-level scope: `clusterer` would
    // then live until the end of main(), i.e. *after* cudaStreamDestroy,
    // and its destructor would submit a free onto an already-destroyed
    // stream (use-after-free -> segfault on exit, even though fit() itself
    // succeeded and everything printed correctly).
    bool ok = false;
    {
        TopologyClusterer clusterer(stream, cfg);
        clusterer.fit(d_features, kSamples, kFeatures);

        const auto& result  = clusterer.result();
        const auto& metrics = clusterer.metrics();

        std::printf("=== TopologyClusterer::fit() smoke test ===\n");
        std::printf("n_samples:        %d (expected %d)\n", result.n_samples, kSamples);
        std::printf("n_clusters found: %d (expected %d)\n", result.n_clusters, kBlobs);
        std::printf("n_noise:          %d\n", result.n_noise);
        std::printf("fit() calls:      %llu\n",
                     static_cast<unsigned long long>(metrics.n_fit_calls.load()));
        std::printf("last fit() time:  %.3f ms\n",
                     static_cast<double>(metrics.total_fit_ns.load()) / 1e6);

        ok = (result.n_samples == kSamples) &&
             (result.n_clusters == kBlobs) &&
             (result.n_noise < kSamples / 10); // allow some noise slack
    } // <-- ~TopologyClusterer() runs here, stream still valid

    cudaFreeAsync(d_features, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (!ok) {
        std::fprintf(stderr, "\nSMOKE TEST FAILED: cluster/noise counts outside expected range.\n");
        return 1;
    }
    std::printf("\nSMOKE TEST PASSED.\n");
    return 0;
}
