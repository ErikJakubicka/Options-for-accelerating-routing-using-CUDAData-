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
#include <thread>
#include <fstream>

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

#define ROOT_SIZE 2048 // 11-bit stride (32KB) for storage in fast Shared Memory

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

// Extracts a specified number of bits from a given position of a 128-bit IPv6 address
uint32_t get_bits(const uint128& ip, int start_bit, int len) {
    uint32_t result = 0;
    for (int i = 0; i < len; i++) {
        int current_bit = start_bit + i;
        int word_idx = current_bit / 32;
        int bit_in_word = 31 - (current_bit % 32);
        
        if ((ip.parts[word_idx] >> bit_in_word) & 1) {
            result |= (1 << (len - 1 - i));
        }
    }
    return result;
}

// Converts an IPv6 address in string format (e.g., "2001:db8::1") into a 128-bit structure
uint128 ip6_to_uint128(const std::string& ip_str) {
    uint128 res = {0, 0, 0, 0};
    if (inet_pton(AF_INET6, ip_str.c_str(), &res.parts) != 1) {
        return res; 
    }
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

// Builds a search tree (Trie) for IPv6 with stride levels split into 11-8-8-8-8-8 bits
void build_ipv6_trie(std::vector<TrieNode>& table, std::vector<BGPRecordV6>& records) {
    table.clear();
    table.resize(ROOT_SIZE, { 0, -1, 1, 0 });

    std::sort(records.begin(), records.end(), [](const BGPRecordV6& a, const BGPRecordV6& b) {
        return a.mask < b.mask;
    });

    for (auto& rec : records) {
        uint32_t l1_idx = get_bits(rec.ip, 0, 11);
        if (rec.mask <= 11) {
            int num = 1 << (11 - rec.mask);
            for (int i = 0; i < num; i++) { table[l1_idx + i].port = rec.port; table[l1_idx + i].is_leaf = 1; }
        } else {
            if (table[l1_idx].is_leaf) {
                int op = table[l1_idx].port; table[l1_idx].is_leaf = 0;
                table[l1_idx].child_index = (int)table.size();
                table.resize(table.size() + 256, { 0, op, 1, 0 });
            }
            uint32_t l2_idx = get_bits(rec.ip, 11, 8);
            int b2 = table[l1_idx].child_index;
            if (rec.mask <= 19) {
                int num = 1 << (19 - rec.mask);
                for (int i = 0; i < num; i++) { table[b2 + l2_idx + i].port = rec.port; table[b2 + l2_idx + i].is_leaf = 1; }
            } else {
                if (table[b2 + l2_idx].is_leaf) {
                    int op = table[b2 + l2_idx].port; table[b2 + l2_idx].is_leaf = 0;
                    table[b2 + l2_idx].child_index = (int)table.size();
                    table.resize(table.size() + 256, { 0, op, 1, 0 });
                }
                uint32_t l3_idx = get_bits(rec.ip, 19, 8);
                int b3 = table[b2 + l2_idx].child_index;
                if (rec.mask <= 27) {
                    int num = 1 << (27 - rec.mask);
                    for (int i = 0; i < num; i++) { table[b3 + l3_idx + i].port = rec.port; table[b3 + l3_idx + i].is_leaf = 1; }
                } else {
                    if (table[b3 + l3_idx].is_leaf) {
                        int op = table[b3 + l3_idx].port; table[b3 + l3_idx].is_leaf = 0;
                        table[b3 + l3_idx].child_index = (int)table.size();
                        table.resize(table.size() + 256, { 0, op, 1, 0 });
                    }
                    uint32_t l4_idx = get_bits(rec.ip, 27, 8);
                    int b4 = table[b3 + l3_idx].child_index;
                    if (rec.mask <= 35) {
                        int num = 1 << (35 - rec.mask);
                        for (int i = 0; i < num; i++) { table[b4 + l4_idx + i].port = rec.port; table[b4 + l4_idx + i].is_leaf = 1; }
                    } else {
                        if (table[b4 + l4_idx].is_leaf) {
                            int op = table[b4 + l4_idx].port; table[b4 + l4_idx].is_leaf = 0;
                            table[b4 + l4_idx].child_index = (int)table.size();
                            table.resize(table.size() + 256, { 0, op, 1, 0 });
                        }
                        uint32_t l5_idx = get_bits(rec.ip, 35, 8);
                        int b5 = table[b4 + l4_idx].child_index;
                        if (rec.mask <= 43) {
                            int num = 1 << (43 - rec.mask);
                            for (int i = 0; i < num; i++) { table[b5 + l5_idx + i].port = rec.port; table[b5 + l5_idx + i].is_leaf = 1; }
                        } else {
                            if (table[b5 + l5_idx].is_leaf) {
                                int op = table[b5 + l5_idx].port; table[b5 + l5_idx].is_leaf = 0;
                                table[b5 + l5_idx].child_index = (int)table.size();
                                table.resize(table.size() + 256, { 0, op, 1, 0 });
                            }
                            uint32_t l6_idx = get_bits(rec.ip, 43, 8);
                            int b6 = table[b5 + l5_idx].child_index;
                            int num = 1 << (std::min(51, rec.mask) - 43);
                            for (int i = 0; i < num; i++) { table[b6 + l6_idx + i].port = rec.port; table[b6 + l6_idx + i].is_leaf = 1; }
                        }
                    }
                }
            }
        }
    }
}

// CUDA Kernel for finding the Longest Prefix Match (LPM) for IPv6 utilizing fast Shared Memory for the 1st level
__global__ void lpm_ipv6_shared_kernel(const uint128* __restrict__ packets, int* __restrict__ results, const TrieNode* __restrict__ global_trie, int batch_size) {
    extern __shared__ TrieNode shared_root[];
    int tid = threadIdx.x;

    for (int i = tid; i < ROOT_SIZE; i += blockDim.x) {
        shared_root[i] = global_trie[i];
    }
    __syncthreads();

    int id = blockIdx.x * blockDim.x + tid;
    if (id < batch_size) {
        uint4 raw_ip = reinterpret_cast<const uint4*>(packets)[id];
        uint128 ip;
        ip.parts[0] = raw_ip.x; ip.parts[1] = raw_ip.y;
        ip.parts[2] = raw_ip.z; ip.parts[3] = raw_ip.w;
        
        TrieNode node = shared_root[(ip.parts[0] >> 21) & 0x7FF];

        if (!node.is_leaf) {
            node = global_trie[node.child_index + ((ip.parts[0] >> 13) & 0xFF)];
            if (!node.is_leaf) {
                node = global_trie[node.child_index + ((ip.parts[0] >> 5) & 0xFF)];
                if (!node.is_leaf) {
                    uint32_t idx4 = ((ip.parts[0] & 0x1F) << 3) | (ip.parts[1] >> 29);
                    node = global_trie[node.child_index + idx4];
                    if (!node.is_leaf) {
                        node = global_trie[node.child_index + ((ip.parts[1] >> 21) & 0xFF)];
                        if (!node.is_leaf) {
                            node = global_trie[node.child_index + ((ip.parts[1] >> 13) & 0xFF)];
                            results[id] = node.port;
                        } else results[id] = node.port;
                    } else results[id] = node.port;
                } else results[id] = node.port;
            } else results[id] = node.port;
        } else results[id] = node.port;
    }
}

// Reference implementation of IPv6 LPM search on the processor (CPU) with OpenMP parallelization support
template <typename AllocIn, typename AllocOut>
void run_ipv6_lookup_cpu(const std::vector<uint128, AllocIn>& packets, int count, int n_threads, const std::vector<TrieNode>& table, std::vector<int, AllocOut>& results) {
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < count; i++) {
        uint128 ip = packets[i];
        
        uint32_t idx1 = (ip.parts[0] >> 21) & 0x7FF; 
        TrieNode node = table[idx1];
        int last_found_port = node.port;

        if (!node.is_leaf) {
            node = table[node.child_index + ((ip.parts[0] >> 13) & 0xFF)];
            if (node.port != -1) last_found_port = node.port;

            if (!node.is_leaf) {
                node = table[node.child_index + ((ip.parts[0] >> 5) & 0xFF)];
                if (node.port != -1) last_found_port = node.port;

                if (!node.is_leaf) {
                    uint32_t idx4 = ((ip.parts[0] & 0x1F) << 3) | (ip.parts[1] >> 29);
                    node = table[node.child_index + idx4];
                    if (node.port != -1) last_found_port = node.port;

                    if (!node.is_leaf) {
                        node = table[node.child_index + ((ip.parts[1] >> 21) & 0xFF)];
                        if (node.port != -1) last_found_port = node.port;

                        if (!node.is_leaf) {
                            node = table[node.child_index + ((ip.parts[1] >> 13) & 0xFF)];
                            if (node.port != -1) last_found_port = node.port;
                        }
                    }
                }
            }
        }
        results[i] = last_found_port;
    }
}

// Loads IPv6 prefixes from a file and simulates assigning an output port
void load_ipv6_records(const std::string& filename, std::vector<BGPRecordV6>& records) {
    std::ifstream file(filename);
    if (!file.is_open()) return;

    std::string line;
    const int NUM_PORTS = 48;
    std::vector<int> port_stats(NUM_PORTS + 1, 0);

    while (std::getline(file, line)) {
        size_t slash = line.find('/');
        if (slash != std::string::npos) {
            std::string ip_part = line.substr(0, slash);
            int mask_val = std::stoi(line.substr(slash + 1));
            
            uint128 ip_val = ip6_to_uint128(ip_part);
            if (ip_val.parts[0] == 0 && ip_val.parts[1] == 0) continue; 
            
            int assigned_port = ((ip_val.parts[0] ^ ip_val.parts[1]) % NUM_PORTS) + 1;

            records.push_back({ip_val, mask_val, assigned_port});
            port_stats[assigned_port]++;
        }
    }

    std::cout << "\n[IPv6 PORT ASSIGNMENT STATISTICS]" << std::endl;
    for (int i = 1; i <= NUM_PORTS; i++) {
        std::cout << " Port " << std::setw(2) << i << ":" << std::setw(5) << port_stats[i];
        if (i % 6 == 0) std::cout << std::endl; else std::cout << " | ";
    }
}

// Runs a measurement of raw GPU kernel performance (without PCIe influence) with various packet batch sizes
template <typename Alloc>
void run_ipv6_sweep_test(TrieNode* d_trie, const std::vector<uint128, Alloc>& host_packets, int max_batch_size, const std::string& filename) {
    const size_t shared_size = ROOT_SIZE * sizeof(TrieNode);
    
    const int warm_batch = 1000000;
    uint128 *d_p_w; int *d_r_w;
    CUDA_CHECK(cudaMalloc(&d_p_w, warm_batch * sizeof(uint128)));
    CUDA_CHECK(cudaMalloc(&d_r_w, warm_batch * sizeof(int)));
    
    lpm_ipv6_shared_kernel<<<(warm_batch + 255) / 256, 256, shared_size>>>(d_p_w, d_r_w, d_trie, warm_batch);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_p_w)); 
    CUDA_CHECK(cudaFree(d_r_w));
    
    std::vector<int> test_sizes = { 1024, 8192, 65536, 524288, 1048576, 5000000, 10000000 };
    
    std::cout << "\n=== SWEEP TEST: IPv6 800G LINE RATE COMPLIANCE ===" << std::endl;
    std::cout << "Batch Size | Time (ms) | Throughput (Mpps) | Status (800G)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int size : test_sizes) {
        if (size > max_batch_size) {
            std::cout << std::setw(10) << size << " | (Exceeds max_batch_size)      | SKIPPED " << std::endl;
            continue;
        }

        int *d_res; uint128 *d_p;
        CUDA_CHECK(cudaMalloc(&d_res, size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_p, size * sizeof(uint128)));
        CUDA_CHECK(cudaMemcpy(d_p, host_packets.data(), size * sizeof(uint128), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop; 
        CUDA_CHECK(cudaEventCreate(&start)); 
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        lpm_ipv6_shared_kernel<<< (size + 255) / 256, 256, shared_size >>>(d_p, d_res, d_trie, size);
        CUDA_CHECK_KERNEL();
        
        CUDA_CHECK(cudaEventRecord(stop)); 
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float mpps = (size / (ms / 1000.0f)) / 1000000.0f;
        std::string status = (mpps >= 1190.0f) ? "OK" : "FAIL";

        std::cout << std::setw(10) << size << " | " << std::setw(9) << ms << " | " << std::setw(17) << mpps << " | " << status << std::endl;
        std::string row = std::to_string(size) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + (mpps >= 1190 ? "OK" : "FAIL");
        log_to_csv(filename, row);
        
        CUDA_CHECK(cudaFree(d_res)); 
        CUDA_CHECK(cudaFree(d_p));
        CUDA_CHECK(cudaEventDestroy(start)); 
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

// Runs CPU computation scalability measurement for different thread counts
template <typename Alloc>
void run_ipv6_cpu_scalability_test(const std::vector<uint128, Alloc>& host_packets, const std::vector<TrieNode>& host_table, const std::string& filename) {
    std::vector<int> thread_counts = { 1, 2, 4, 8, 16 };
    const int BATCH_SIZE = (int)host_packets.size();
    std::vector<int> results(BATCH_SIZE);

    std::cout << "\n=== IPv6 CPU SCALABILITY TEST (AMD Ryzen 7 3700X) ===" << std::endl;
    std::cout << " Threads | Time (ms) | Throughput (Mpps) | Latency (ns) " << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    run_ipv6_lookup_cpu(host_packets, BATCH_SIZE, 16, host_table, results);

    for (int tc : thread_counts) {
        auto start = std::chrono::high_resolution_clock::now();
        
        run_ipv6_lookup_cpu(host_packets, BATCH_SIZE, tc, host_table, results);
        
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(end - start).count();
        float mpps = (BATCH_SIZE / (ms / 1000.0f)) / 1000000.0f;
        float lat = (ms * 1000000.0f) / BATCH_SIZE;

        std::cout << std::setw(8) << tc << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(9) << ms << " | " 
                  << std::setprecision(2) << std::setw(17) << mpps << " | "
                  << std::setw(12) << lat << std::endl;
                  std::string row = std::to_string(tc) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + std::to_string(lat);
                  log_to_csv(filename, row);
    }
    std::cout << "========================================================\n" << std::endl;
}

// Measures the efficiency of asynchronous overlapping of transfers and computations using various amounts of CUDA streams
template <typename Alloc>
void run_ipv6_gpu_stream_test(TrieNode* d_trie, const std::vector<uint128, Alloc>& host_packets, const std::string& filename) {
    std::vector<int> stream_counts = { 2, 3, 4 };
    const int BATCH_SIZE = (int)host_packets.size();
    const size_t shared_size = ROOT_SIZE * sizeof(TrieNode); 
    
    uint128 *d_p; int *d_r;
    CUDA_CHECK(cudaMalloc(&d_p, BATCH_SIZE * sizeof(uint128)));
    CUDA_CHECK(cudaMalloc(&d_r, BATCH_SIZE * sizeof(int)));
    std::vector<int, CudaPinnedAllocator<int>> final_res(BATCH_SIZE);

    std::cout << "\n=== IPv6 GPU STREAM TEST (Shared Memory + Streams) ===" << std::endl;
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
        lpm_ipv6_shared_kernel<<<warm_blocks, warm_threads, shared_size, streams[0]>>>(d_p, d_r, d_trie, chunk_size);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ns; i++) {
            int offset = i * chunk_size;
            CUDA_CHECK(cudaMemcpyAsync(d_p + offset, host_packets.data() + offset, chunk_size * sizeof(uint128), cudaMemcpyHostToDevice, streams[i]));
            
            int threads = 256;
            int blocks = (chunk_size + threads - 1) / threads;
            lpm_ipv6_shared_kernel<<<blocks, threads, shared_size, streams[i]>>>(d_p + offset, d_r + offset, d_trie, chunk_size);
            CUDA_CHECK_KERNEL();
            
            CUDA_CHECK(cudaMemcpyAsync(final_res.data() + offset, d_r + offset, chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float ms; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float mpps = (BATCH_SIZE / (ms / 1000.0f)) / 1000000.0f;

        std::cout << std::setw(8) << ns << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(9) << ms << " | " 
                  << std::setprecision(2) << std::setw(17) << mpps << " | "
                  << (mpps >= 1190.0f ? "OK" : "FAIL") << std::endl;
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
    // === A. PREPARATION AND DATA LOADING ===
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    std::vector<BGPRecordV6> records;
    load_ipv6_records("ipv6_prefixes.txt", records);
    if (records.empty()) {
        uint128 test_ip = { 0x20010db8, 0, 0, 0 };
        records.push_back({ test_ip, 32, 10 });
    }

    std::vector<TrieNode> host_table;
    host_table.reserve(100 * 1024 * 1024);
    build_ipv6_trie(host_table, records);

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
    for (int i = 0; i < BATCH_SIZE; i++) {
        host_packets[i].parts[0] = 0x20010db8;
        host_packets[i].parts[1] = rand();
        host_packets[i].parts[2] = rand();
        host_packets[i].parts[3] = rand();
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

    // === G. CONFIGURING SHARED MEMORY AND STREAMS ===
    CUDA_CHECK(cudaFuncSetAttribute(lpm_ipv6_shared_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(lpm_ipv6_shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));

    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int chunk_size = BATCH_SIZE / num_streams;
    size_t shared_size = ROOT_SIZE * sizeof(TrieNode);
    std::vector<int, CudaPinnedAllocator<int>> gpu_final_res(BATCH_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); 
    CUDA_CHECK(cudaEventCreate(&stop));

    // === H. RUNNING SYSTEM MEASUREMENT AND STRESS TEST (GPU) ===
    int threads_per_block = 256;
    int blocks_per_grid = (chunk_size + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemcpyAsync(d_packets, host_packets.data(), chunk_size * sizeof(uint128), cudaMemcpyHostToDevice, streams[0]));
    lpm_ipv6_shared_kernel<<<blocks_per_grid, threads_per_block, shared_size, streams[0]>>>(d_packets, d_results, d_trie, chunk_size);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start)); 

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(d_packets + offset, host_packets.data() + offset, chunk_size * sizeof(uint128), cudaMemcpyHostToDevice, streams[i]));
            
        lpm_ipv6_shared_kernel << <(chunk_size + 255) / 256, 256, shared_size, streams[i] >> > (d_packets + offset, d_results + offset, d_trie, chunk_size);
        CUDA_CHECK_KERNEL();
            
        CUDA_CHECK(cudaMemcpyAsync(gpu_final_res.data() + offset, d_results + offset, chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop)); 
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // === I. COMPARISON WITH CPU AND CPU STRESS TEST === 
    std::vector<int> cpu_results(BATCH_SIZE);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    run_ipv6_lookup_cpu(host_packets, BATCH_SIZE, 16, host_table, cpu_results); 
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(end_cpu - cpu_start).count();

    // === J. STATISTIC CALCULATIONS AND OUTPUTS === 
    const float GPU_TDP = 215.0f; 
    const float CPU_TDP = 65.0f;  

    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f;
    float cpu_mpps = (BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f;

    float gpu_eff = gpu_mpps / GPU_TDP;
    float cpu_eff = cpu_mpps / CPU_TDP;

    std::cout << "  IPv6 SHARED MEMORY            " << std::endl;
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

    std::string final_row = "IPv6_Shared;" + 
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