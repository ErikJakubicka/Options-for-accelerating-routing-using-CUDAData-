#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iomanip>
#include <fstream>
#include <winsock2.h>  
#include <ws2tcpip.h>  
#include <chrono>
#include <omp.h>
#include <thread> 

#pragma comment(lib, "ws2_32.lib")

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

// Structure representing a single node in the search tree (Trie), aligned to 16 bytes for consistency and performance
struct alignas(16) TrieNode {
    int child_index; 
    int port;
    int is_leaf;
    int padding;
};

// Structure to store a 128-bit IPv6 address in the form of four 32-bit parts
struct alignas(16) uint128 {
    uint32_t parts[4]; 
};

// Helper structure to store a parsed record from the IPv6 routing table
struct BGPRecordV6 {
    uint128 ip;
    int mask;
    int port;
};

// Extracts one specific bit from a 128-bit IPv6 address based on the given position
__host__ __device__ inline int get_bit_128(const uint128& ip, int bit_pos) {
    int word_idx = 3 - (bit_pos / 32);
    int bit_in_word = bit_pos % 32;
    return (ip.parts[word_idx] >> bit_in_word) & 1;
}

// Converts an IPv6 address in string format (e.g., "2001:db8::1") into a 128-bit structure
uint128 ip6_to_uint128(const std::string& ip_str) {
    uint128 res = {0, 0, 0, 0};
    if (inet_pton(AF_INET6, ip_str.c_str(), &res.parts) != 1) return res;
    for(int i = 0; i < 4; i++) {
        uint32_t v = res.parts[i];
        res.parts[i] = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v >> 8) & 0xFF00) | (v >> 24);
    }
    return res;
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

// Builds a Single-bit search tree (Trie) for IPv6, where one memory jump is made for each bit of the address (depth up to 128)
void build_ipv6_single_bit_trie(std::vector<TrieNode>& table, std::vector<BGPRecordV6>& records) {
    table.clear();
    table.push_back({-1, -1, 0, 0}); 

    std::sort(records.begin(), records.end(), [](const BGPRecordV6& a, const BGPRecordV6& b) {
        return a.mask < b.mask;
    });

    for (auto& rec : records) {
        int current_idx = 0;
        for (int i = 127; i >= (128 - rec.mask); --i) {
            int bit = get_bit_128(rec.ip, i);
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

// CUDA Kernel for naive bit-by-bit Longest Prefix Match (LPM) search for IPv6 in global GPU memory
__global__ void lpm_ipv6_single_bit_kernel(const uint128* __restrict__ packets, int* __restrict__ results, const TrieNode* __restrict__ trie, int batch_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        uint4 raw_ip = reinterpret_cast<const uint4*>(packets)[id];
        uint128 ip;
        ip.parts[0] = raw_ip.x; ip.parts[1] = raw_ip.y;
        ip.parts[2] = raw_ip.z; ip.parts[3] = raw_ip.w;
        int current_idx = 0;
        int last_found_port = -1;

        for (int i = 127; i >= 0; --i) {
            TrieNode node = trie[current_idx];
            if (node.port != -1) last_found_port = node.port;
            
            if (node.is_leaf || node.child_index <= 0) break;

            int bit = get_bit_128(ip, i);
            current_idx = node.child_index + bit;
        }
        results[id] = last_found_port;
    }
}

// Reference implementation of IPv6 Single-bit LPM search on the processor (CPU) with OpenMP parallelization support
template <typename AllocIn, typename AllocOut>
void lpm_ipv6_single_bit_cpu(const std::vector<uint128, AllocIn>& packets, int count, int n_threads, const std::vector<TrieNode>& table, std::vector<int, AllocOut>& results) {
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < count; i++) {
        uint128 ip = packets[i];
        int current_idx = 0;
        int last_found_port = -1;
        for (int b = 127; b >= 0; b--) {
            TrieNode node = table[current_idx];
            if (node.port != -1) last_found_port = node.port;
            if (node.is_leaf || node.child_index <= 0) break;
            current_idx = node.child_index + get_bit_128(ip, b);
        }
        results[i] = last_found_port;
    }
}

// Runs a measurement of raw GPU kernel performance (without PCIe influence) with various packet batch sizes
template <typename Alloc>
void run_ipv6_sweep_test(TrieNode* d_trie, const std::vector<uint128, Alloc>& host_packets, int max_batch_size, const std::string& filename) {
    const int warm_batch = 1000000;
    uint128 *d_p_w; int *d_r_w;
    CUDA_CHECK(cudaMalloc(&d_p_w, warm_batch * sizeof(uint128)));
    CUDA_CHECK(cudaMalloc(&d_r_w, warm_batch * sizeof(int)));
    
    lpm_ipv6_single_bit_kernel<<<(warm_batch + 255) / 256, 256>>>(d_p_w, d_r_w, d_trie, warm_batch);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_p_w)); 
    CUDA_CHECK(cudaFree(d_r_w));

    std::vector<int> test_sizes = { 1024, 8192, 65536, 524288, 1048576, 5000000, 10000000 };
    std::cout << "\n=== BASELINE IPv6 SWEEP TEST: SINGLE-BIT PERFORMANCE ===" << std::endl;
    std::cout << "Batch Size | Time (ms) | Throughput (Mpps) | Status (800G)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int size : test_sizes) {
        if (size > max_batch_size) continue;
        uint128 *d_p; int *d_r;
        CUDA_CHECK(cudaMalloc(&d_p, size * sizeof(uint128))); 
        CUDA_CHECK(cudaMalloc(&d_r, size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_p, host_packets.data(), size * sizeof(uint128), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop; 
        CUDA_CHECK(cudaEventCreate(&start)); 
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        lpm_ipv6_single_bit_kernel<<<(size + 255) / 256, 256>>>(d_p, d_r, d_trie, size);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float mpps = (size / (ms / 1000.0f)) / 1000000.0f;
        std::cout << std::setw(10) << size << " | " << std::fixed << std::setprecision(4) << std::setw(9) << ms 
                  << " | " << std::setw(17) << mpps << " | " << (mpps >= 1190 ? "OK" : "FAIL") << std::endl;
                  std::string row = std::to_string(size) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + (mpps >= 1190 ? "OK" : "FAIL");
                  log_to_csv(filename, row);

        CUDA_CHECK(cudaFree(d_p)); 
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    std::cout << "====================================================" << std::endl;
}

// Runs CPU computation scalability measurement for different thread counts
template <typename Alloc>
void run_ipv6_cpu_scalability_test(const std::vector<uint128, Alloc>& host_packets, const std::vector<TrieNode>& host_table, const std::string& filename) {
    std::vector<int> thread_counts = { 1, 2, 4, 8, 16 };
    const int BATCH_SIZE = (int)host_packets.size();
    std::vector<int> results(BATCH_SIZE);
    
    std::cout << "\n=== BASELINE IPv6 CPU SCALABILITY (Single-bit) ===" << std::endl;
    std::cout << " Threads | Time (ms) | Throughput (Mpps) | Latency (ns) " << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    lpm_ipv6_single_bit_cpu(host_packets, BATCH_SIZE, 16, host_table, results);

    for (int tc : thread_counts) {
        auto start = std::chrono::high_resolution_clock::now();
        lpm_ipv6_single_bit_cpu(host_packets, BATCH_SIZE, tc, host_table, results);
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();
        float mpps = (BATCH_SIZE / (ms / 1000.0f)) / 1000000.0f;
        float lat = (ms * 1000000.0f) / BATCH_SIZE;

        std::cout << std::setw(8) << tc << " | " << std::fixed << std::setprecision(4) << std::setw(9) << ms 
                  << " | " << std::setprecision(2) << std::setw(17) << mpps << " | " << std::setw(12) << lat << std::endl;
                  std::string row = std::to_string(tc) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + std::to_string(lat);
                  log_to_csv(filename, row);
    }
    std::cout << "====================================================" << std::endl;
}

// Measures the efficiency of asynchronous overlapping of transfers and computations using various amounts of CUDA streams
template <typename Alloc>
void run_ipv6_gpu_stream_test(TrieNode* d_trie, const std::vector<uint128, Alloc>& host_packets, const std::string& filename) {
    std::vector<int> stream_counts = { 2, 3, 4 };
    const int BATCH_SIZE = (int)host_packets.size();
    uint128 *d_p; int *d_r;
    CUDA_CHECK(cudaMalloc(&d_p, BATCH_SIZE * sizeof(uint128))); 
    CUDA_CHECK(cudaMalloc(&d_r, BATCH_SIZE * sizeof(int)));
    std::vector<int, CudaPinnedAllocator<int>> final_res(BATCH_SIZE);

    std::cout << "\n=== BASELINE IPv6 GPU STREAM TEST (Single-bit Pipelining) ===" << std::endl;
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
        
        CUDA_CHECK(cudaMemcpyAsync(d_p, host_packets.data(), chunk_size * sizeof(uint128), cudaMemcpyHostToDevice, streams[0]));
        lpm_ipv6_single_bit_kernel<<<warm_blocks, warm_threads, 0, streams[0]>>>(d_p, d_r, d_trie, chunk_size);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ns; i++) {
            int offset = i * chunk_size;
            CUDA_CHECK(cudaMemcpyAsync(d_p + offset, host_packets.data() + offset, chunk_size * sizeof(uint128), cudaMemcpyHostToDevice, streams[i]));
            
            lpm_ipv6_single_bit_kernel<<<(chunk_size + 255) / 256, 256, 0, streams[i]>>>(d_p + offset, d_r + offset, d_trie, chunk_size);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpyAsync(final_res.data() + offset, d_r + offset, chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
        }
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaDeviceSynchronize());
        
        float ms; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float mpps = (BATCH_SIZE / (ms / 1000.0f)) / 1000000.0f;
        std::cout << std::setw(8) << ns << " | " << std::fixed << std::setprecision(4) << std::setw(9) << ms 
                  << " | " << std::setprecision(2) << std::setw(17) << mpps << " | " << (mpps >= 1190 ? "OK" : "FAIL") << std::endl;
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
    std::cout << "====================================================" << std::endl;
}

int main() {
    // === A. PREPARATION AND DATA LOADING ===
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
    std::vector<BGPRecordV6> records;
    std::ifstream file("ipv6_prefixes.txt");
    std::string line;
    while (std::getline(file, line)) {
        size_t slash = line.find('/');
        if (slash != std::string::npos) {
            records.push_back({ip6_to_uint128(line.substr(0, slash)), std::stoi(line.substr(slash + 1)), (int)(records.size() % 48 + 1)});
        }
    }
    if(records.empty()) records.push_back({{0x20010db8, 0, 0, 0}, 32, 10});

    std::vector<TrieNode> host_table;
    build_ipv6_single_bit_trie(host_table, records);

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
        std::ofstream f4(final_file); f4 << "Type;GPU_Ms;GPU_Mpps;GPU_Eff;CPU_Ms;CPU_Mpps;CPU_Eff;Speedup;Improvement\n";
    }

    // === B. TEST PACKET GENERATION ===
    const int BATCH_SIZE = 1000000;
    std::vector<uint128, CudaPinnedAllocator<uint128>> host_packets(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++) { 
        host_packets[i].parts[0] = 0x20010db8; 
        host_packets[i].parts[1] = rand(); 
    }

    // === C. GPU ALLOCATION AND TREE TRANSFER ===
    TrieNode *d_trie; uint128 *d_packets; int *d_results;
    CUDA_CHECK(cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode)));
    CUDA_CHECK(cudaMalloc(&d_packets, BATCH_SIZE * sizeof(uint128)));
    CUDA_CHECK(cudaMalloc(&d_results, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice));

    // === D. RUNNING SWEEP TEST ===
    run_ipv6_sweep_test(d_trie, host_packets, BATCH_SIZE, sweep_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // === E. RUNNING GPU STREAM TEST ===
    run_ipv6_gpu_stream_test(d_trie, host_packets, stream_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // === F. RUNNING CPU SCALABILITY TEST ===
    run_ipv6_cpu_scalability_test(host_packets, host_table, cpu_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // === G. SETTING UP STREAMS AND EVENTS FOR ASYNCHRONOUS EXECUTION ===
    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    int chunk_size = BATCH_SIZE / num_streams;
    std::vector<int, CudaPinnedAllocator<int>> gpu_final_res(BATCH_SIZE);

    // FINAL END-TO-END MEASUREMENT
    cudaEvent_t start, stop; 
    CUDA_CHECK(cudaEventCreate(&start)); 
    CUDA_CHECK(cudaEventCreate(&stop));

    // === H. RUNNING SYSTEM MEASUREMENT AND STRESS TEST (GPU) ===
    int threads_per_block = 256;
    int blocks_per_grid = (chunk_size + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemcpyAsync(d_packets, host_packets.data(), chunk_size * sizeof(uint128), cudaMemcpyHostToDevice, streams[0]));
    lpm_ipv6_single_bit_kernel<<<blocks_per_grid, threads_per_block, 0, streams[0]>>>(d_packets, d_results, d_trie, chunk_size);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start)); 

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;

        CUDA_CHECK(cudaMemcpyAsync(d_packets + offset, host_packets.data() + offset, 
                        chunk_size * sizeof(uint128), cudaMemcpyHostToDevice, streams[i]));

        lpm_ipv6_single_bit_kernel<<<(chunk_size + 255) / 256, 256, 0, streams[i]>>>(
            d_packets + offset, d_results + offset, d_trie, chunk_size);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaMemcpyAsync(gpu_final_res.data() + offset, d_results + offset, 
                        chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop)); 
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_ms = 0; 
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // === I. COMPARISON WITH CPU AND CPU STRESS TEST ===
    std::vector<int> cpu_results(BATCH_SIZE);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    lpm_ipv6_single_bit_cpu(host_packets, BATCH_SIZE, 16, host_table, cpu_results); 
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(end_cpu - cpu_start).count();

    // === J. STATISTIC CALCULATIONS AND OUTPUTS ===
    const float GPU_TDP = 215.0f; 
    const float CPU_TDP = 65.0f;  

    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f;
    float cpu_mpps = (BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f;

    float gpu_eff = gpu_mpps / GPU_TDP;
    float cpu_eff = cpu_mpps / CPU_TDP;

    std::cout << "  IPv6 SINGLE BIT        " << std::endl;
    std::cout << "\n  Packets: " << BATCH_SIZE << " | Prefixes: " << records.size() << std::endl;
    std::cout << "  HW: NVIDIA RTX 2070 SUPER | CUDA Streams: 2" << std::endl;
    std::cout << "====================================================" << std::endl;

    std::cout << "\n GPU RESULTS (RTX 2070 SUPER - End-to-End):" << std::endl;
    std::cout << "  Time (Transfer+Kernel):  " << std::fixed << std::setprecision(2) << gpu_ms << " ms" << std::endl;
    std::cout << "  Throughput:           " << gpu_mpps << " Mpps" << std::endl;
    std::cout << "  Energy efficiency: " << gpu_eff << " Mpk/J" << std::endl;

    std::cout << "\n CPU RESULTS (Single-Core Reference):" << std::endl;
    std::cout << "  CPU Time (16 threads): " << cpu_ms << " ms" << std::endl;
    std::cout << "  Throughput:            " << cpu_mpps << " Mpps" << std::endl;
    std::cout << "  Energy efficiency:  " << cpu_eff << " Mpk/J" << std::endl;

    std::cout << "\n COMPARISON (SPEEDUP):" << std::endl;
    std::cout << "  GPU is " << (gpu_mpps / cpu_mpps) << "x faster than CPU." << std::endl;
    std::cout << "  GPU is " << (gpu_eff / cpu_eff) << "x more energy efficient." << std::endl;
    std::cout << "====================================================" << std::endl;

    std::string final_row = "IPv6_Single;" + 
                            std::to_string(gpu_ms) + ";" + std::to_string(gpu_mpps) + ";" + std::to_string(gpu_eff) + ";" + 
                            std::to_string(cpu_ms) + ";" + std::to_string(cpu_mpps) + ";" + std::to_string(cpu_eff) + ";" + 
                            std::to_string(gpu_mpps / cpu_mpps) + ";" + std::to_string(gpu_eff / cpu_eff);
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
    WSACleanup();
    
    return 0;
}