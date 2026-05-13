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

#include <thrust/device_vector.h> 
#include <thrust/sort.h>          
#include <thrust/execution_policy.h>

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

// Macro for error checking immediately after the kernel starts
#define CUDA_CHECK_KERNEL() \
do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error when launching a kernel in " << __FILE__ << " on line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// A custom allocator for allocating pinned memory on the CPU, which speeds up PCIe transfers
template <typename T>
struct CudaPinnedAllocator {
    using value_type = T;

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        cudaError_t err = cudaMallocHost((void**)&ptr, n * sizeof(T));
        if (err != cudaSuccess) throw std::bad_alloc();
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        CUDA_CHECK(cudaFreeHost(ptr)); 
    }
};

// A structure representing a single node in a search tree (Trie), aligned to 16 bytes for the Texture Cache
struct alignas(16) TrieNode {
    int child_index; 
    int port;        
    int is_leaf;     
    int padding;     
};

// Helper structure for storing a parsed entry from the routing table
struct BGPRecord {
    uint32_t ip; 
    int mask;    
    int port;    
};

// Converts an IP address in string format to a 32-bit number
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

// Builds a routing tree (Trie) with 16-8-8-bit subnets based on BGP records
void build_real_trie_16_8_8(std::vector<TrieNode>& table, std::vector<BGPRecord>& records) {
    table.clear();
    table.resize(65536, {0, -1, 1, 0}); 

    std::sort(records.begin(), records.end(), [](const BGPRecord& a, const BGPRecord& b) {
        return a.mask < b.mask;
    });

    for (auto& rec : records) {
        if (rec.mask <= 16) {
            uint32_t startIdx = rec.ip >> 16;
            int numBlocks = 1 << (16 - rec.mask); 
            for (int i = 0; i < numBlocks; i++) { 
                table[startIdx + i].port = rec.port; 
                table[startIdx + i].is_leaf = 1; 
            }
        } else if (rec.mask <= 24) {
            uint32_t l1 = (rec.ip >> 16) & 0xFFFF;
            if (table[l1].is_leaf) {
                int op = table[l1].port; table[l1].is_leaf = 0;
                table[l1].child_index = (int)table.size(); 
                table.resize(table.size() + 256, {0, op, 1, 0}); 
            }
            uint32_t l2_s = (rec.ip >> 8) & 0xFF; 
            int num = 1 << (24 - rec.mask);
            int base = table[l1].child_index;
            for (int i = 0; i < num; i++) { 
                table[base + l2_s + i].port = rec.port; 
                table[base + l2_s + i].is_leaf = 1; 
            }
        } else {
            uint32_t l1 = (rec.ip >> 16) & 0xFFFF;
            if (table[l1].is_leaf) {
                int op = table[l1].port; table[l1].is_leaf = 0;
                table[l1].child_index = (int)table.size(); 
                table.resize(table.size() + 256, {0, op, 1, 0});
            }
            uint32_t l2 = (rec.ip >> 8) & 0xFF;
            int b2 = table[l1].child_index;
            if (table[b2 + l2].is_leaf) {
                int op = table[b2 + l2].port; table[b2 + l2].is_leaf = 0;
                table[b2 + l2].child_index = (int)table.size(); 
                table.resize(table.size() + 256, {0, op, 1, 0});
            }
            uint32_t l3_s = rec.ip & 0xFF; 
            int num = 1 << (32 - rec.mask);
            int b3 = table[b2 + l2].child_index;
            for (int i = 0; i < num; i++) { 
                table[b3 + l3_s + i].port = rec.port; 
                table[b3 + l3_s + i].is_leaf = 1; 
            }
        }
    }
}

// CUDA kernel for finding the longest matching pattern (LPM) using fast texture memory (Texture Cache) on the GPU
__global__ void lpm_final_kernel(const unsigned int* __restrict__ packets, int* __restrict__ results, cudaTextureObject_t trieTex, int batch_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (id < batch_size) {
        unsigned int ip = packets[id];
        
        int4 raw = tex1Dfetch<int4>(trieTex, (ip >> 16) & 0xFFFF);
        TrieNode node = *(reinterpret_cast<TrieNode*>(&raw));
        
        if (!node.is_leaf) {
            raw = tex1Dfetch<int4>(trieTex, node.child_index + ((ip >> 8) & 0xFF));
            TrieNode node2 = *(reinterpret_cast<TrieNode*>(&raw));
            if (!node2.is_leaf) {
                raw = tex1Dfetch<int4>(trieTex, node2.child_index + (ip & 0xFF));
                TrieNode node3 = *(reinterpret_cast<TrieNode*>(&raw));
                results[id] = node3.port;
            } else results[id] = node2.port;
        } else results[id] = node.port;
    }
}

// Reference implementation of LPM search on a central processing unit (CPU) with support for OpenMP parallelization
template <typename AllocIn, typename AllocOut>
void lpm_cpu_reference(const std::vector<unsigned int, AllocIn>& packets, int n_threads, const std::vector<TrieNode>& table, std::vector<int, AllocOut>& results) {
    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < (int)packets.size(); ++i) {
        unsigned int ip = packets[i];
        TrieNode node = table[(ip >> 16) & 0xFFFF];
        if (!node.is_leaf) {
            TrieNode node2 = table[node.child_index + ((ip >> 8) & 0xFF)];
            if (!node2.is_leaf) results[i] = table[node2.child_index + (ip & 0xFF)].port;
            else results[i] = node2.port;
        } else results[i] = node.port;
    }
}

// Reads prefixes from a file and simulates output port mapping based on the first octet of the IP address
void load_records_from_file(const std::string& filename, std::vector<BGPRecord>& records) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        return;
    }

    std::string line;
    const int NUM_PORTS = 48; 
    std::vector<int> port_stats(NUM_PORTS + 1, 0); 

    while (std::getline(file, line)) {
        size_t slash = line.find('/');
        if (slash != std::string::npos) {
            uint32_t ip_val = ip_to_uint(line.substr(0, slash));
            int mask_val = std::stoi(line.substr(slash + 1));
            
            uint8_t first_octet = (ip_val >> 24) & 0xFF;
            int assigned_port = (first_octet % NUM_PORTS) + 1;

            records.push_back({ip_val, mask_val, assigned_port});
            port_stats[assigned_port]++;
        }
    }

    std::cout << "\n[PORT ASSIGNMENT STATISTICS (" << NUM_PORTS << " ports)]" << std::endl;
    for (int i = 1; i <= NUM_PORTS; i++) {
        std::cout << " Port " << std::setw(2) << i << ": " << std::setw(6) << port_stats[i] << " prefixes";
        if (i % 3 == 0) std::cout << std::endl; else std::cout << " | ";
    }
    std::cout << std::endl;
}

// Saves the final (IP address, Port) pairs to a CSV file (e.g., for verification purposes)
void save_results_to_file(const std::string& filename, const std::vector<unsigned int>& packets, const std::vector<int>& results) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;
    outfile << "IP_ADDRESS,OUTPUT_PORT\n";
    for (size_t i = 0; i < packets.size(); ++i) {
        unsigned int ip = packets[i];
        outfile << ((ip >> 24) & 0xFF) << "." << ((ip >> 16) & 0xFF) << "." << ((ip >> 8) & 0xFF) << "." << (ip & 0xFF) 
                << "," << results[i] << "\n";
    }
}

// Runs a measurement of the GPU kernel's raw performance (without the influence of the PCIe bus) using texture memory
template <typename Alloc>
void run_sweep_test(TrieNode* d_trie, cudaTextureObject_t trieTex, const std::vector<unsigned int, Alloc>& host_packets, const std::string& filename) {
    const int warm_batch = 1000000;
    unsigned int *d_p_w; int *d_r_w;
    CUDA_CHECK(cudaMalloc(&d_p_w, warm_batch * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_r_w, warm_batch * sizeof(int)));
    
    lpm_final_kernel<<<(warm_batch + 255) / 256, 256>>>(d_p_w, d_r_w, trieTex, warm_batch);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaFree(d_p_w)); 
    CUDA_CHECK(cudaFree(d_r_w));
    
    std::vector<int> test_sizes = { 1024, 8192, 65536, 524288, 1048576, 5000000, 10000000 };
    
    std::cout << "\n=== SWEEP TEST: TEXTURE CACHE (RANDOM ACCESS) ===" << std::endl;
    std::cout << "Batch Size | Time (ms) | Throughput (Mpps) | Status (800G)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int size : test_sizes) {
        if (size > host_packets.size()) {
          std::cout << std::setw(10) << size << " | (Exceeds max_batch_size)      | SKIPPED " << std::endl;
          continue;
        }
        unsigned int *d_p; int *d_r;
        CUDA_CHECK(cudaMalloc(&d_p, size * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_r, size * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_p, host_packets.data(), size * sizeof(unsigned int), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); 
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        lpm_final_kernel<<<(size + 255)/256, 256>>>(d_p, d_r, trieTex, size);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float mpps = (size / (ms / 1000.0f)) / 1000000.0f;
        
        std::cout << std::setw(10) << size << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(9) << ms << " | " 
                  << std::setw(17) << mpps << " | "
                  << (mpps > 1190 ? "OK" : "FAIL") << std::endl;
                  std::string row = std::to_string(size) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + (mpps >= 1190 ? "OK" : "FAIL");
                  log_to_csv(filename, row);

        CUDA_CHECK(cudaFree(d_p)); 
        CUDA_CHECK(cudaFree(d_r));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    std::cout << "============================================================\n" << std::endl;
}

// Runs a measurement of CPU computation scalability for different numbers of threads
template <typename Alloc>
void run_cpu_scalability_test(const std::vector<unsigned int, Alloc>& host_packets, const std::vector<TrieNode>& host_table, const std::string& filename) {
    std::vector<int> thread_counts = { 1, 2, 4, 8, 16 };
    const int BATCH_SIZE = (int)host_packets.size();
    std::vector<int> results(BATCH_SIZE);

    std::cout << "\n=== CPU SCALABILITY TEST (AMD Ryzen 7 3700X) ===" << std::endl;
    std::cout << " Threads | Time (ms) | Throughput (Mpps) | Latency (ns) " << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    lpm_cpu_reference(host_packets, 16, host_table, results);

    for (int tc : thread_counts) {
        auto start = std::chrono::high_resolution_clock::now();
        
        lpm_cpu_reference(host_packets, tc, host_table, results);
        
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

// Measures the efficiency of asynchronous interleaving of transfers and computations using different numbers of CUDA streams
template <typename Alloc>
void run_gpu_stream_test(cudaTextureObject_t trieTex, const std::vector<unsigned int, Alloc>& host_packets, const std::string& filename) {
    std::vector<int> stream_counts = { 2, 3, 4 };
    const int BATCH_SIZE = (int)host_packets.size();
    
    unsigned int *d_p; int *d_r;
    CUDA_CHECK(cudaMalloc(&d_p, BATCH_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_r, BATCH_SIZE * sizeof(int)));
    std::vector<int, CudaPinnedAllocator<int>> final_res(BATCH_SIZE);

    std::cout << "\n=== GPU STREAM TEST (Pipelining Efficiency) ===" << std::endl;
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
        lpm_final_kernel<<<warm_blocks, warm_threads, 0, streams[0]>>>(d_p, d_r, trieTex, chunk_size);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ns; i++) {
            int offset = i * chunk_size;
            CUDA_CHECK(cudaMemcpyAsync(d_p + offset, host_packets.data() + offset, chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]));
            
            int threads = 256;
            int blocks = (chunk_size + threads - 1) / threads;
            lpm_final_kernel<<<blocks, threads, 0, streams[i]>>>(d_p + offset, d_r + offset, trieTex, chunk_size);
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
                  << (mpps > 1190 ? "OK" : "FAIL") << std::endl;
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
    // === A. LOADING AND DATA PREPARATION === 
    std::vector<BGPRecord> records;
    load_records_from_file("unique_prefixes.txt", records);
    if (records.empty()) {
        std::cerr << "Error: No records in unique_prefixes.txt" << std::endl;
        return 1;
    }

    std::vector<TrieNode> host_table;
    host_table.reserve(10 * 1024 * 1024); 
    build_real_trie_16_8_8(host_table, records);

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

    // === B. GENERATING TEST PACKETS === 
    const int BATCH_SIZE = 10000000; 
    std::vector<unsigned int, CudaPinnedAllocator<unsigned int>> host_packets(BATCH_SIZE);

    for(int i = 0; i < BATCH_SIZE; i++) {
        host_packets[i] = ((unsigned int)rand() << 16) | (rand() & 0xFFFF);
    }

    // === C. GPU ALLOCATION AND TREE TRANSFER ===
    TrieNode *d_trie; 
    int *d_results;
    unsigned int *d_packets_raw;

    CUDA_CHECK(cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode)));
    CUDA_CHECK(cudaMalloc(&d_results, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_packets_raw, BATCH_SIZE * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice));

    // === D. SETTING UP TEXTURE OBJECT (GPU Cache) ===
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_trie;
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    resDesc.res.linear.sizeInBytes = host_table.size() * sizeof(TrieNode);
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t trieTex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&trieTex, &resDesc, &texDesc, NULL));

    // === E. RUNNING SWEEP TEST ===
    run_sweep_test(d_trie, trieTex, host_packets, sweep_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // === F. RUNNING GPU STREAM TEST ===
    run_gpu_stream_test(trieTex, host_packets, stream_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // === G. RUNNING CPU SCALABILITY TEST ===
    run_cpu_scalability_test(host_packets, host_table, cpu_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // === H. SETTING UP STREAMS AND EVENTS FOR ASYNCHRONOUS EXECUTION ===
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

    // === I. RUNNING SYSTEM MEASUREMENT AND STRESS TEST (GPU) ===
    int threads_per_block = 256;
    int blocks_per_grid = (chunk_size + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemcpyAsync(d_packets_raw, host_packets.data(), chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[0]));
    lpm_final_kernel<<<blocks_per_grid, threads_per_block, 0, streams[0]>>>(d_packets_raw, d_results, trieTex, chunk_size);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start)); 
    
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;

        CUDA_CHECK(cudaMemcpyAsync(d_packets_raw + offset, host_packets.data() + offset, 
                        chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]));

        int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        lpm_final_kernel<<<blocks, threads, 0, streams[i]>>>(d_packets_raw + offset, d_results + offset, trieTex, chunk_size);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK(cudaMemcpyAsync(final_res.data() + offset, d_results + offset, 
                        chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop)); 
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float gpu_ms = 0; 
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    // === J. COMPARISON WITH CPU AND CPU STRESS TEST ===
    std::vector<int> cpu_res(BATCH_SIZE);
    
    auto cpu_s = std::chrono::high_resolution_clock::now();
    lpm_cpu_reference(host_packets, 16, host_table, cpu_res);
    auto cpu_e = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_e - cpu_s).count();

    // === K. CALCULATING STATISTICS AND OUTPUTS ===
    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f;
    float cpu_mpps = (BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f;
    float lat_per_pkt = (gpu_ms * 1000000.0f) / BATCH_SIZE;
    float cpu_lat_per_pkt = (cpu_ms * 1000000.0f) / BATCH_SIZE;
    float gpu_eff = gpu_mpps / 215.0f; 
    float cpu_eff = ((BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f) / 65.0f;

    std::cout << "    IPv4 TEXTURE CACHE - END-TO-END BENCHMARK    " << std::endl;
    std::cout << "\n  Packets: " << BATCH_SIZE << " | Prefixes: " << records.size() << std::endl;
    std::cout << "   HW: NVIDIA RTX 2070 SUPER | CUDA Streams: " << num_streams << std::endl;
    std::cout << "====================================================" << std::endl;

    std::cout << "\nGPU SYSTEM MEASUREMENT RESULT:" << std::endl;
    std::cout << "  Average system latency: " << std::fixed << std::setprecision(2) << lat_per_pkt << " ns" << std::endl;
    std::cout << "  Time (Transfer + Kernel):      " << gpu_ms << " ms" << std::endl;
    std::cout << "  System throughput:       " << gpu_mpps << " Mpps" << std::endl;

    std::cout << "\nENERGY EFFICIENCY (Mpk/J):" << std::endl;
    std::cout << "  GPU (System-wide): " << gpu_eff << std::endl; 
    std::cout << "  CPU (System-wide): " << cpu_eff << std::endl;
    std::cout << "  Improvement:          " << gpu_eff / cpu_eff << "x" << std::endl;

    std::cout << "\nCOMPARISON WITH CPU:" << std::endl;
    std::cout << "  16-thread CPU: " << cpu_ms << " ms" << std::endl;
    std::cout << "  Speedup:     " << cpu_ms / gpu_ms << "x" << std::endl;
    std::cout << "====================================================" << std::endl;

    std::string final_row = "IPv4_Texture;" + std::to_string(lat_per_pkt) + ";" + std::to_string(gpu_ms) + ";" + std::to_string(gpu_mpps) + ";" + 
                            std::to_string(gpu_eff) + ";" + std::to_string(cpu_eff) + ";" + std::to_string(gpu_eff / cpu_eff) + ";" + std::to_string(cpu_ms) + ";" + std::to_string(cpu_ms / gpu_ms);
    log_to_csv(final_file, final_row);

    // === L. MEMORY CLEANUP ===
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaDestroyTextureObject(trieTex));
    CUDA_CHECK(cudaFree(d_trie)); 
    CUDA_CHECK(cudaFree(d_results)); 
    CUDA_CHECK(cudaFree(d_packets_raw));
    
    return 0;
}