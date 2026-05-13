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

#pragma comment(lib, "ws2_32.lib") // Nutné pre Windows

#define ROOT_SIZE 2048 // 11-bit stride (32KB), aby sme mali rezervu v Shared Memory

struct alignas(16) TrieNode {
    int child_index;
    int port;
    int is_leaf;
    int padding;
};

struct uint128 {
    uint32_t parts[4]; // parts[0] je MSB (najvýznamnejších 32 bitov)
};

struct BGPRecordV6 {
    uint128 ip;
    int mask;
    int port;
};

// 2. POMOCNÉ FUNKCIE
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

// Prevod textovej IPv6 na našu uint128 štruktúru
uint128 ip6_to_uint128(const std::string& ip_str) {
    uint128 res = {0, 0, 0, 0};
    // inet_pton zapíše 16 bajtov v sieťovom poradí (Big-Endian)
    if (inet_pton(AF_INET6, ip_str.c_str(), &res.parts) != 1) {
        return res; 
    }
    // Musíme prehodiť bajty, ak naša get_bits počíta s Host Byte Order (Little-Endian na x86)
    for(int i=0; i<4; i++) {
        uint32_t v = res.parts[i];
        res.parts[i] = ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v >> 8) & 0xFF00) | (v >> 24);
    }
    return res;
}

// 3. BUILDER (STRIDE 11 - 8 - 8 - 8 - 8 - 8)
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
            // LEVEL 2
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
                // LEVEL 3
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
                    // LEVEL 4
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
                        // LEVEL 5
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
                            // LEVEL 6 (až do /51)
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

// 4. CUDA KERNEL (SHARED MEMORY)
__global__ void lpm_ipv6_shared_kernel(uint128* packets, int* results, TrieNode* global_trie, int batch_size) {
    extern __shared__ TrieNode shared_root[];
    int tid = threadIdx.x;

    for (int i = tid; i < ROOT_SIZE; i += blockDim.x) {
        shared_root[i] = global_trie[i];
    }
    __syncthreads();

    int id = blockIdx.x * blockDim.x + tid;
    if (id < batch_size) {
        uint128 ip = packets[id];
        
        // L1 (Shared Memory)
        TrieNode node = shared_root[(ip.parts[0] >> 21) & 0x7FF];

        if (!node.is_leaf) {
            // L2 (Global Memory)
            node = global_trie[node.child_index + ((ip.parts[0] >> 13) & 0xFF)];
            
            if (!node.is_leaf) {
                // L3
                node = global_trie[node.child_index + ((ip.parts[0] >> 5) & 0xFF)];

                if (!node.is_leaf) {
                    // L4 (Bit-Stitching)
                    uint32_t idx4 = ((ip.parts[0] & 0x1F) << 3) | (ip.parts[1] >> 29);
                    node = global_trie[node.child_index + idx4];

                    if (!node.is_leaf) {
                        // L5
                        node = global_trie[node.child_index + ((ip.parts[1] >> 21) & 0xFF)];

                        if (!node.is_leaf) {
                            // L6 (Konečne sme pri /48 - /51)
                            node = global_trie[node.child_index + ((ip.parts[1] >> 13) & 0xFF)];
                            results[id] = node.port;
                        } else results[id] = node.port;
                    } else results[id] = node.port;
                } else results[id] = node.port;
            } else results[id] = node.port;
        } else results[id] = node.port;
    }
}


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
            if (ip_val.parts[0] == 0 && ip_val.parts[1] == 0) continue; // Preskoč neplatné/prázdne
            
            // Port pridelíme podľa prvého segmentu IPv6 (v parts[0])
            int assigned_port = ((ip_val.parts[0] ^ ip_val.parts[1]) % NUM_PORTS) + 1;

            records.push_back({ip_val, mask_val, assigned_port});
            port_stats[assigned_port]++;
        }
    }

    std::cout << "\n[STATISTIKA PRIRADENIA IPv6 PORTOV]" << std::endl;
    for (int i = 1; i <= NUM_PORTS; i++) {
        std::cout << " Port " << std::setw(2) << i << ":" << std::setw(5) << port_stats[i];
        if (i % 6 == 0) std::cout << std::endl; else std::cout << " | ";
    }
}

// --- SWEEP TEST ---
void run_ipv6_sweep_test(TrieNode* d_trie, const std::vector<uint128>& host_packets, int max_batch_size) {
    std::vector<int> test_sizes = { 1024, 8192, 65536, 524288, 1048576, 10000000 };
    
    std::cout << "\n=== SWEEP TEST: IPv6 800G LINE RATE COMPLIANCE ===" << std::endl;
    std::cout << "Batch Size | Time (ms) | Throughput (Mpps) | Status (800G)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int size : test_sizes) {
        // --- OCHRANA: Overenie Batch Size ---
        if (size > max_batch_size) {
            std::cout << std::setw(10) << size << " | (Exceeds max_batch_size)      | SKIPPED " << std::endl;
            continue;
        }

        int *d_res; uint128 *d_p;
        cudaMalloc(&d_res, size * sizeof(int));
        cudaMalloc(&d_p, size * sizeof(uint128));
        cudaMemcpy(d_p, host_packets.data(), size * sizeof(uint128), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        size_t shared_size = ROOT_SIZE * sizeof(TrieNode);
        lpm_ipv6_shared_kernel<<< (size + 255) / 256, 256, shared_size >>>(d_p, d_res, d_trie, size);
        
        cudaEventRecord(stop); cudaDeviceSynchronize();

        float ms; cudaEventElapsedTime(&ms, start, stop);
        float mpps = (size / (ms / 1000.0f)) / 1000000.0f;
        std::string status = (mpps >= 1190.0f) ? "OK" : "FAIL";

        std::cout << std::setw(10) << size << " | " << std::setw(9) << ms << " | " << std::setw(17) << mpps << " | " << status << std::endl;
        
        cudaFree(d_res); cudaFree(d_p);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
}

void run_ipv6_lookup_cpu(const std::vector<TrieNode>& table, const std::vector<uint128>& packets, std::vector<int>& results) {
    for (size_t i = 0; i < packets.size(); i++) {
        uint128 ip = packets[i];
        
        // L1: Root (vždy začíname tu)
        uint32_t idx1 = (ip.parts[0] >> 21) & 0x7FF; 
        TrieNode node = table[idx1];
        int last_found_port = node.port;

        // Ak nie je list, ideme do L2
        if (!node.is_leaf) {
            node = table[node.child_index + ((ip.parts[0] >> 13) & 0xFF)];
            if (node.port != -1) last_found_port = node.port;

            // Ak nie je list, ideme do L3
            if (!node.is_leaf) {
                node = table[node.child_index + ((ip.parts[0] >> 5) & 0xFF)];
                if (node.port != -1) last_found_port = node.port;

                // Ak nie je list, ideme do L4
                if (!node.is_leaf) {
                    uint32_t idx4 = ((ip.parts[0] & 0x1F) << 3) | (ip.parts[1] >> 29);
                    node = table[node.child_index + idx4];
                    if (node.port != -1) last_found_port = node.port;

                    // Ak nie je list, ideme do L5
                    if (!node.is_leaf) {
                        node = table[node.child_index + ((ip.parts[1] >> 21) & 0xFF)];
                        if (node.port != -1) last_found_port = node.port;

                        // Ak nie je list, ideme do L6
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

int main() {
    // A. PRÍPRAVA (WinSock a Načítanie PREFIXOV)
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    std::vector<BGPRecordV6> records;
    // ZMENA: Načítavame testovacie prefixy pre stavbu stromu
    load_ipv6_records("test_prefixes.txt", records);
    if (records.empty()) {
        std::cerr << "Chyba: test_prefixes.txt nenajdeny!" << std::endl;
        return -1;
    }

    std::vector<TrieNode> host_table;
    host_table.reserve(1024 * 1024); // Pre test stačí malá rezerva
    build_ipv6_trie(host_table, records);

    // --- B. NAČÍTANIE TESTOVACÍCH PAKETOV ---
    std::vector<uint128> host_packets;
    std::vector<std::string> packet_labels; // Pre neskorší výpis
    std::ifstream pkt_file("test_packets.txt");
    std::string line;

    while (std::getline(pkt_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        host_packets.push_back(ip6_to_uint128(line));
        packet_labels.push_back(line);
    }

    const int BATCH_SIZE = (int)host_packets.size();
    if (BATCH_SIZE == 0) {
        std::cerr << "Chyba: test_packets.txt je prazdny!" << std::endl;
        return -1;
    }

    // C. GPU ALOKÁCIA
    TrieNode *d_trie; uint128 *d_packets; int *d_results;
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode));
    cudaMalloc(&d_packets, BATCH_SIZE * sizeof(uint128));
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));

    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packets, host_packets.data(), BATCH_SIZE * sizeof(uint128), cudaMemcpyHostToDevice);

    // D. CONFIG GPU
    cudaFuncSetAttribute(lpm_ipv6_shared_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(lpm_ipv6_shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

    // E. VÝPOČET (Pre verifikáciu stačí 1 stream, ale nechávame logiku pre konzistenciu)
    size_t shared_size = ROOT_SIZE * sizeof(TrieNode);
    std::vector<int> gpu_final_res(BATCH_SIZE);

    // Spustíme kernel (pre pár paketov stačí 1 blok)
    lpm_ipv6_shared_kernel<<<1, 256, shared_size>>>(d_packets, d_results, d_trie, BATCH_SIZE);
    
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_final_res.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // F. CPU VÝPOČET PRE KONTROLU
    std::vector<int> cpu_results(BATCH_SIZE);
    run_ipv6_lookup_cpu(host_table, host_packets, cpu_results);

    // --- G. VERIFIKAČNÝ REPORT ---
    std::cout << "\n====================================================" << std::endl;
    std::cout << "          IPv6 VERIFICATION REPORT (LPM)            " << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << std::left << std::setw(35) << "Testovana IPv6 adresa" 
              << " | " << std::setw(10) << "GPU Port" 
              << " | " << std::setw(10) << "CPU Port" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::string gpu_out = (gpu_final_res[i] <= 0) ? "DROPPED" : std::to_string(gpu_final_res[i]);
        std::string cpu_out = (cpu_results[i] <= 0) ? "DROPPED" : std::to_string(cpu_results[i]);

        std::cout << std::left << std::setw(35) << packet_labels[i] 
                  << " | " << std::setw(10) << gpu_out 
                  << " | " << std::setw(10) << cpu_out;
        
        if (gpu_final_res[i] == cpu_results[i]) {
            std::cout << " [ MATCH ]" << std::endl;
        } else {
            std::cout << " [ ERROR! ]" << std::endl;
        }
    }
    std::cout << "====================================================" << std::endl;

    // ČISTENIE
    cudaFree(d_trie); cudaFree(d_packets); cudaFree(d_results);
    WSACleanup();
    return 0;
}