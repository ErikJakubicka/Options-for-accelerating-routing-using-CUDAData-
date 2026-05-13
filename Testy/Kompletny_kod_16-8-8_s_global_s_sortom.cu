#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h> 
#include <chrono>         
#include <fstream>        
#include <iomanip>

#include <thrust/device_vector.h> 
#include <thrust/sort.h>          
#include <thrust/execution_policy.h>

// 1. ŠTRUKTÚRA UZLA (Zarovnaná na 16B pre maximálnu priepustnosť zbernice)
struct alignas(16) TrieNode {
    int child_index; 
    int port;        
    int is_leaf;     
    int padding;     
};

struct BGPRecord {
    uint32_t ip; 
    int mask;    
    int port;    
};

uint32_t ip_to_uint(const std::string& ip_str) {
    unsigned int a, b, c, d;
    if (sscanf(ip_str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) return 0;
    return (a << 24) | (b << 16) | (c << 8) | d;
}

// 2. BUILDER 16-8-8 (Logika Longest Prefix Match)
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

// 3. CUDA KERNEL (GLOBAL MEMORY + SORTED ACCESS)
__global__ void lpm_global_kernel(unsigned int* packets, int* results, TrieNode* trie, int batch_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; 
    if (id < batch_size) {
        unsigned int ip = packets[id];
        
        // Priamy prístup do Global Memory. 
        // Vďaka predošlému 'thrust::sort' bude tento prístup COALESCED (zlúčený).
        TrieNode node = trie[(ip >> 16) & 0xFFFF];
        
        if (!node.is_leaf) {
            TrieNode node2 = trie[node.child_index + ((ip >> 8) & 0xFF)];
            if (!node2.is_leaf) {
                results[id] = trie[node2.child_index + (ip & 0xFF)].port;
            } else results[id] = node2.port;
        } else results[id] = node.port;
    }
}

// 4. NAČÍTANIE S VÝPISOM ŠTATISTIKY
void load_records_from_file(const std::string& filename, std::vector<BGPRecord>& records) {
    std::ifstream file(filename);
    if (!file.is_open()) return;

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

    std::cout << "\n[STATISTIKA PRIRADENIA PORTOV (" << NUM_PORTS << " portov)]" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    for (int i = 1; i <= NUM_PORTS; i++) {
        std::cout << " Port " << std::setw(2) << i << ": " << std::setw(6) << port_stats[i] << " prefixov";
        if (i % 3 == 0) std::cout << std::endl; else std::cout << " | ";
    }
    std::cout << "----------------------------------------------------" << std::endl;
}

// CPU REFERENCE
void lpm_cpu_reference(const std::vector<unsigned int>& packets, const std::vector<TrieNode>& table, std::vector<int>& results) {
    for (size_t i = 0; i < packets.size(); ++i) {
        unsigned int ip = packets[i];
        TrieNode node = table[(ip >> 16) & 0xFFFF];
        if (!node.is_leaf) {
            TrieNode node2 = table[node.child_index + ((ip >> 8) & 0xFF)];
            if (!node2.is_leaf) results[i] = table[node2.child_index + (ip & 0xFF)].port;
            else results[i] = node2.port;
        } else results[i] = node.port;
    }
}

// HLAVNÝ PROGRAM
int main() {
    // A. PRÍPRAVA
    std::vector<BGPRecord> records;
    load_records_from_file("unique_prefixes.txt", records);
    if (records.empty()) return 1;

    std::vector<TrieNode> host_table;
    host_table.reserve(10 * 1024 * 1024); 
    build_real_trie_16_8_8(host_table, records);

    const int BATCH_SIZE = 10000;
    std::vector<unsigned int> host_packets(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++) {
        host_packets[i] = ((unsigned int)rand() << 16) | (rand() & 0xFFFF);
    }

    // B. GPU ALOKÁCIA
    TrieNode *d_trie; int *d_results;
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode)); 
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));         
    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);

    // C. OPTIMALIZÁCIA: RADENIE (THRUST)
    auto t_sort_start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<unsigned int> d_packets_vec = host_packets;
    thrust::sort(d_packets_vec.begin(), d_packets_vec.end());
    auto t_sort_end = std::chrono::high_resolution_clock::now();
    float sort_ms = std::chrono::duration<float, std::milli>(t_sort_end - t_sort_start).count();

    unsigned int* d_packets_ptr = thrust::raw_pointer_cast(d_packets_vec.data());

    // D. SPUSTENIE (Streams + Global Memory)
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1); cudaStreamCreate(&s2);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int threads = 256; 
    int half = BATCH_SIZE / 2;
    int blocks = (half + threads - 1) / threads; 

    cudaEventRecord(start); 
    lpm_global_kernel<<<blocks, threads, 0, s1>>>(d_packets_ptr, d_results, d_trie, half);
    lpm_global_kernel<<<blocks, threads, 0, s2>>>(d_packets_ptr + half, d_results + half, d_trie, half);
    cudaEventRecord(stop); 
    
    cudaDeviceSynchronize(); 
    float gpu_ms = 0; cudaEventElapsedTime(&gpu_ms, start, stop);

    // E. ANALÝZA
    auto cpu_s = std::chrono::high_resolution_clock::now();
    std::vector<int> cpu_res(BATCH_SIZE);
    lpm_cpu_reference(host_packets, host_table, cpu_res);
    auto cpu_e = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_e - cpu_s).count();

    // F. ŠTATISTIKY
    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f; 
    float gpu_eff = gpu_mpps / 215.0f; 
    float cpu_eff = ((BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f) / 65.0f;

    // VÝPISY NA KONZOLU
    std::cout << "\n====================================================" << std::endl;
    std::cout << "GLOBAL MEMORY + THRUST SORT (COALESCED)" << std::endl;
    std::cout << "  Pakety: " << BATCH_SIZE << " | Prefixy: " << records.size() << std::endl;
    std::cout << "  HW: NVIDIA RTX 2070 SUPER | CUDA Streams: 2" << std::endl;
    std::cout << "====================================================" << std::endl;

    std::cout << "\nVYSLEDOK MERANIA:" << std::endl;
    std::cout << "  Cas radenia (Thrust): " << sort_ms << " ms" << std::endl;
    std::cout << "  Cas Lookupu (GPU):    " << gpu_ms << " ms" << std::endl;
    std::cout << "  Priepustnost:         " << gpu_mpps << " Mpps" << std::endl;

    std::cout << "\nENERGETICKA EFEKTIVITA (Mpk/J):" << std::endl;
    std::cout << "  GPU: " << gpu_eff << " | CPU: " << cpu_eff << std::endl;
    std::cout << "  Zlepsenie: " << gpu_eff / cpu_eff << "x" << std::endl;

    std::cout << "\nPOROVNANIE:" << std::endl;
    std::cout << "  Cas na CPU: " << cpu_ms << " ms" << std::endl;
    std::cout << "  Zrychlenie: " << cpu_ms / gpu_ms << "x" << std::endl;
    std::cout << "====================================================" << std::endl;

    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFree(d_trie); cudaFree(d_results);
    return 0;
}