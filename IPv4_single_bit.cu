#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h> 
#include <chrono>         
#include <fstream>        
#include <iomanip>
#include <omp.h>
#include <thread>

// --- CUDA ERROR HANDLING MACROS ---

// Macro for standard CUDA API calls (cudaMalloc, cudaMemcpy, etc.)
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " on line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Macro for checking errors immediately after kernel launch
#define CUDA_CHECK_KERNEL() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error when launching kernel in " << __FILE__ << " on line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// Custom allocator for allocating pinned memory on the CPU, which speeds up PCIe transfers
template <typename T>
struct CudaPinnedAllocator {
    using value_type = T;

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost((void**)&ptr, n * sizeof(T)));
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        CUDA_CHECK(cudaFreeHost(ptr)); 
    }
};

// Structure representing a single node in the search tree (Trie), aligned to 16 bytes for consistency
struct alignas(16) TrieNode {
    int child_index; 
    int port;
    int is_leaf;
    int padding;
};

// Helper structure to store a parsed routing table record
struct BGPRecord {
    uint32_t ip;
    int mask;
    int port;
};

// Converts an IP address in string format to a 32-bit integer
uint32_t ip_to_uint(const std::string& ip_str) {
    unsigned int a, b, c, d;
    if (sscanf(ip_str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) return 0;
    return (a << 24) | (b << 16) | (c << 8) | d;
}

// Writes formatted data (logs/results) to a specified CSV file
void log_to_csv(const std::string& filename, const std::string& data) {
    std::ofstream file;
    file.open(filename, std::ios_base::app); 
    if (file.is_open()) {
        file << data << "\n";
        file.close();
    }
}

// Builds a basic Single-bit search tree (Trie), where one jump is made for each bit of the IP address
void build_single_bit_trie(std::vector<TrieNode>& table, std::vector<BGPRecord>& records) {
    table.clear();
    table.push_back({-1, -1, 0, 0}); 

    std::sort(records.begin(), records.end(), [](const BGPRecord& a, const BGPRecord& b) {
        return a.mask < b.mask;
    });

    for (auto& rec : records) {
        int current_idx = 0;
        for (int i = 31; i >= (32 - rec.mask); --i) {
            int bit = (rec.ip >> i) & 1;
            
            if (table[current_idx].child_index <= 0) {
                table[current_idx].child_index = (int)table.size();
                table[current_idx].is_leaf = 0;
                table.push_back({-1, -1, 1, 0}); 
                table.push_back({-1, -1, 1, 0}); 
            }
            current_idx = table[current_idx].child_index + bit;
        }
        table[current_idx].port = rec.port;
        table[current_idx].is_leaf = 1;
    }
}

// CUDA Kernel for naive bit-by-bit Longest Prefix Match (LPM) search with a maximum of 32 memory accesses
__global__ void lpm_single_bit_kernel(const unsigned int* __restrict__ packets, int* __restrict__ results, const TrieNode* __restrict__ trie, int batch_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        unsigned int ip = packets[id];
        int current_idx = 0;
        int last_found_port = -1;

        for (int i = 31; i >= 0; --i) {
            TrieNode node = trie[current_idx];
            if (node.port != -1) last_found_port = node.port;
            
            if (node.is_leaf || node.child_index <= 0) break;

            int bit = (ip >> i) & 1;
            current_idx = node.child_index + bit;
        }
        results[id] = last_found_port;
    }
}

// Reference implementation of Single-bit LPM search on the processor (CPU) with OpenMP parallelization support
template <typename AllocIn, typename AllocOut>
void lpm_single_bit_cpu(const std::vector<unsigned int, AllocIn>& packets, int count, int n_threads, const std::vector<TrieNode>& table, std::vector<int, AllocOut>& results) {
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < count; i++) {
        unsigned int ip = packets[i];
        int current_idx = 0;
        int last_found_port = -1;

        for (int b = 31; b >= 0; b--) {
            TrieNode node = table[current_idx];
            if (node.port != -1) last_found_port = node.port;
            if (node.is_leaf || node.child_index <= 0) break;
            current_idx = node.child_index + ((ip >> b) & 1);
        }
        results[i] = last_found_port;
    }
}

// Runs a measurement of raw GPU kernel performance (without PCIe influence) for various batch sizes
template <typename Alloc>
void run_sweep_test(TrieNode* d_trie, const std::vector<unsigned int, Alloc>& host_packets, const std::string& filename) {
    const int warm_batch = 1000000;
    unsigned int *d_p_w; int *d_r_w;
    CUDA_CHECK(cudaMalloc(&d_p_w, warm_batch * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_r_w, warm_batch * sizeof(int)));
    
    lpm_single_bit_kernel<<<(warm_batch + 255) / 256, 256>>>(d_p_w, d_r_w, d_trie, warm_batch);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_p_w)); 
    CUDA_CHECK(cudaFree(d_r_w));

    std::vector<int> test_sizes = { 1024, 8192, 65536, 524288, 1048576, 5000000, 10000000 };
    std::cout << "\n=== BASELINE SWEEP TEST: SINGLE-BIT KERNEL PERFORMANCE ===" << std::endl;
    std::cout << "Batch Size | Time (ms) | Throughput (Mpps) | Status (800G)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int size : test_sizes) {
        if (size > host_packets.size()) continue;
        unsigned int *d_p; int *d_r;
        CUDA_CHECK(cudaMalloc(&d_p, size * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_r, size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_p, host_packets.data(), size * sizeof(unsigned int), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop; 
        CUDA_CHECK(cudaEventCreate(&start)); 
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        lpm_single_bit_kernel<<<(size + 255)/256, 256>>>(d_p, d_r, d_trie, size);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float mpps = (size / (ms / 1000.0f)) / 1000000.0f;
        std::cout << std::setw(10) << size << " | " << std::fixed << std::setprecision(4) << std::setw(9) << ms 
                  << " | " << std::setw(17) << mpps << " | " << (mpps > 1190 ? "OK" : "FAIL") << std::endl;
                  std::string row = std::to_string(size) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + (mpps >= 1190 ? "OK" : "FAIL");
                  log_to_csv(filename, row);

        CUDA_CHECK(cudaFree(d_p)); 
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    std::cout << "========================================================\n" << std::endl;
}

// Runs CPU computation scalability measurement for different thread counts
template <typename Alloc>
void run_cpu_scalability_test(const std::vector<unsigned int, Alloc>& host_packets, const std::vector<TrieNode>& host_table, const std::string& filename) {
    std::vector<int> thread_counts = { 1, 2, 4, 8, 16 };
    const int BATCH_SIZE = (int)host_packets.size();
    std::vector<int> results(BATCH_SIZE);

    std::cout << "\n=== BASELINE CPU SCALABILITY (Single-bit) ===" << std::endl;
    std::cout << " Threads | Time (ms) | Throughput (Mpps) | Latency (ns) " << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    lpm_single_bit_cpu(host_packets, BATCH_SIZE, 16, host_table, results);

    for (int tc : thread_counts) {
        auto start = std::chrono::high_resolution_clock::now();
        lpm_single_bit_cpu(host_packets, BATCH_SIZE, tc, host_table, results);
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();
        float mpps = (BATCH_SIZE / (ms / 1000.0f)) / 1000000.0f;
        float lat = (ms * 1000000.0f) / BATCH_SIZE;

        std::cout << std::setw(8) << tc << " | " << std::fixed << std::setprecision(4) << std::setw(9) << ms 
                  << " | " << std::setprecision(2) << std::setw(17) << mpps << " | " << std::setw(12) << lat << std::endl;
                  std::string row = std::to_string(tc) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + std::to_string(lat);
                  log_to_csv(filename, row);
    }
    std::cout << "========================================================\n" << std::endl;
}

// Measures the efficiency of asynchronous overlapping of transfers and computations using various amounts of CUDA streams
template <typename Alloc>
void run_gpu_stream_test(TrieNode* d_trie, const std::vector<unsigned int, Alloc>& host_packets, const std::string& filename) {
    std::vector<int> stream_counts = { 2, 3, 4 };
    const int BATCH_SIZE = (int)host_packets.size();
    unsigned int *d_p; int *d_r;
    CUDA_CHECK(cudaMalloc(&d_p, BATCH_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_r, BATCH_SIZE * sizeof(int)));
    std::vector<int, CudaPinnedAllocator<int>> final_res(BATCH_SIZE);

    std::cout << "\n=== BASELINE GPU STREAM TEST (Single-bit Pipelining) ===" << std::endl;
    std::cout << " Streams | Time (ms) | Throughput (Mpps) | Status (800G) " << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int ns : stream_counts) {
        int chunk_size = BATCH_SIZE / ns;
        std::vector<cudaStream_t> streams(ns);
        for (int i = 0; i < ns; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }

        cudaEvent_t start, stop; 
        CUDA_CHECK(cudaEventCreate(&start)); 
        CUDA_CHECK(cudaEventCreate(&stop));

        int warm_threads = 256;
        int warm_blocks = (chunk_size + warm_threads - 1) / warm_threads;
        
        CUDA_CHECK(cudaMemcpyAsync(d_p, host_packets.data(), chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[0]));
        lpm_single_bit_kernel<<<warm_blocks, warm_threads, 0, streams[0]>>>(d_p, d_r, d_trie, chunk_size);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ns; i++) {
            int offset = i * chunk_size;
            CUDA_CHECK(cudaMemcpyAsync(d_p + offset, host_packets.data() + offset, chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]));
            
            lpm_single_bit_kernel<<<(chunk_size + 255) / 256, 256, 0, streams[i]>>>(d_p + offset, d_r + offset, d_trie, chunk_size);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpyAsync(final_res.data() + offset, d_r + offset, chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
        }
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float ms; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float mpps = (BATCH_SIZE / (ms / 1000.0f)) / 1000000.0f;
        std::cout << std::setw(8) << ns << " | " << std::fixed << std::setprecision(4) << std::setw(9) << ms 
                  << " | " << std::setprecision(2) << std::setw(17) << mpps << " | " << (mpps > 1190 ? "OK" : "FAIL") << std::endl;
                  std::string row = std::to_string(ns) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + (mpps >= 1190 ? "OK" : "FAIL");
                  log_to_csv(filename, row);
        
        for (int i = 0; i < ns; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    CUDA_CHECK(cudaFree(d_p)); 
    CUDA_CHECK(cudaFree(d_r));
    std::cout << "========================================================\n" << std::endl;
}

int main() {
    // === A. DATA LOADING AND PREPARATION ===
    std::vector<BGPRecord> records;
    std::ifstream file("unique_prefixes.txt");
    std::string line;
    while (std::getline(file, line)) {
        size_t slash = line.find('/');
        if (slash != std::string::npos) {
            records.push_back({ip_to_uint(line.substr(0, slash)), std::stoi(line.substr(slash + 1)), (int)(records.size() % 48 + 1)});
        }
    }

    std::vector<TrieNode> host_table;
    build_single_bit_trie(host_table, records);

    double memory_footprint_mb = (host_table.size() * sizeof(TrieNode)) / (1024.0 * 1024.0);
    std::cout << "\n[TREE MEMORY FOOTPRINT]" << std::endl;
    std::cout << "  Number of nodes in tree: " << host_table.size() << std::endl;
    std::cout << "  Size of one node: " << sizeof(TrieNode) << " bytes" << std::endl;
    std::cout << "  Total size:      " << std::fixed << std::setprecision(2) << memory_footprint_mb << " MB" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    std::string sweep_file = "test_sweep.csv";
    std::string stream_file = "test_streams.csv";
    std::string cpu_file = "test_cpu.csv";
    std::string final_file = "test_final.csv";

    {
        std::ofstream f1(sweep_file); f1 << "Batch Size;Time (ms);Throughput (Mpps);Status\n";
        std::ofstream f2(stream_file); f2 << "Streams;Time (ms);Throughput (Mpps);Status\n";
        std::ofstream f3(cpu_file); f3 << "Threads;Time (ms);Throughput (Mpps);Latency (ns)\n";
        std::ofstream f4(final_file); f4 << "Type;Lat_per_pkt(ns);GPU_Ms;GPU_Mpps;GPU_Eff;CPU_Eff;Improvement;CPU_Ms;Speedup\n";
    }

    // === B. TEST PACKET GENERATION ===
    const int BATCH_SIZE = 10000000;
    std::vector<unsigned int, CudaPinnedAllocator<unsigned int>> host_packets(BATCH_SIZE);
    for(int i=0; i<BATCH_SIZE; i++) host_packets[i] = rand() | (rand() << 16);

    // === C. GPU ALLOCATION AND TREE TRANSFER ===
    TrieNode *d_trie; unsigned int *d_packets; int *d_results;
    CUDA_CHECK(cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode)));
    CUDA_CHECK(cudaMalloc(&d_packets, BATCH_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_results, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice));

    // === D. RUNNING SWEEP TEST ===
    run_sweep_test(d_trie, host_packets, sweep_file);
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // === E. RUNNING GPU STREAM TEST ===
    run_gpu_stream_test(d_trie, host_packets, stream_file);
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // === F. RUNNING CPU SCALABILITY TEST ===
    run_cpu_scalability_test(host_packets, host_table, cpu_file);
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // === G. SETTING UP STREAMS AND EVENTS FOR ASYNCHRONOUS EXECUTION ===
    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int chunk_size = BATCH_SIZE / num_streams;
    std::vector<int, CudaPinnedAllocator<int>> final_res(BATCH_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); 
    CUDA_CHECK(cudaEventCreate(&stop));

    // === H. RUNNING SYSTEM MEASUREMENT AND STRESS TEST (GPU) ===
    int threads_per_block = 256;
    int blocks_per_grid = (chunk_size + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemcpyAsync(d_packets, host_packets.data(), chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[0]));
    lpm_single_bit_kernel<<<blocks_per_grid, threads_per_block, 0, streams[0]>>>(d_packets, d_results, d_trie, chunk_size);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start)); 

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;

        CUDA_CHECK(cudaMemcpyAsync(d_packets + offset, host_packets.data() + offset, 
                        chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]));

        int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        lpm_single_bit_kernel<<<blocks, threads, 0, streams[i]>>>(d_packets + offset, d_results + offset, d_trie, chunk_size);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaMemcpyAsync(final_res.data() + offset, d_results + offset, 
                        chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop)); 
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_ms = 0; 
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // === I. COMPARISON WITH CPU AND CPU STRESS TEST ===
    std::vector<int> cpu_res(BATCH_SIZE);
    
    auto cpu_s = std::chrono::high_resolution_clock::now();
    lpm_single_bit_cpu(host_packets, BATCH_SIZE, 16, host_table, cpu_res);
    auto cpu_e = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_e - cpu_s).count();

    // === J. STATISTIC CALCULATIONS AND OUTPUTS ===
    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f;
    float gpu_ns_per_pkt = (gpu_ms * 1000000.0f) / BATCH_SIZE;
    float gpu_eff = gpu_mpps / 215.0f; 
    float cpu_eff = ((BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f) / 65.0f;

    std::cout << "  IPv4 SINGLE BIT - END-TO-END BENCHMARK    " << std::endl;
    std::cout << "\n  Packets: " << BATCH_SIZE << " | Prefixes: " << records.size() << std::endl;
    std::cout << "  HW: NVIDIA RTX 2070 SUPER  "<< std::endl;
    std::cout << "====================================================" << std::endl;

    std::cout << "\nGPU SYSTEM MEASUREMENT RESULT:" << std::endl;
    std::cout << "  Average system latency: " << gpu_ns_per_pkt << " ns" << std::endl;
    std::cout << "  Time (Transfer + Kernel):      " << std::fixed << std::setprecision(4) << gpu_ms << " ms" << std::endl;
    std::cout << "  System throughput:       " << std::setprecision(2) << gpu_mpps << " Mpps" << std::endl;

    std::cout << "\nENERGY EFFICIENCY (Mpk/J):" << std::endl;
    std::cout << "  GPU (System-wide): " << gpu_eff << std::endl; 
    std::cout << "  CPU (System-wide): " << cpu_eff << std::endl;
    std::cout << "  Improvement:         " << gpu_eff / cpu_eff << "x" << std::endl;

    std::cout << "\nCOMPARISON WITH CPU:" << std::endl;
    std::cout << "  16-thread CPU: " << std::fixed << std::setprecision(4) << cpu_ms << " ms" << std::endl;;
    std::cout << "  Speedup:     " << cpu_ms / gpu_ms << "x" << std::endl;
    std::cout << "====================================================" << std::endl;

    std::string final_row = "IPv4_Global_Single;" + std::to_string(gpu_ns_per_pkt) + ";" + std::to_string(gpu_ms) + ";" + std::to_string(gpu_mpps) + ";" + 
                            std::to_string(gpu_eff) + ";" + std::to_string(cpu_eff) + ";" + std::to_string(gpu_eff / cpu_eff) + ";" + std::to_string(cpu_ms) + ";" + std::to_string(cpu_ms / gpu_ms);
    log_to_csv(final_file, final_row);

    // === K. MEMORY CLEANUP ===
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_trie)); 
    CUDA_CHECK(cudaFree(d_packets)); 
    CUDA_CHECK(cudaFree(d_results));
    
    return 0;
}