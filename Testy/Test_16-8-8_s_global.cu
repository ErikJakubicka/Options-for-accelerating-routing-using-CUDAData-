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

// 1. ŠTRUKTÚRA UZLA (16B)
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

// 2. BUILDER 16-8-8
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

// 3. CUDA KERNEL (GLOBAL MEMORY + RANDOM ACCESS)
__global__ void lpm_global_kernel(unsigned int* packets, int* results, TrieNode* trie, int batch_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        unsigned int ip = packets[id];
        TrieNode node = trie[(ip >> 16) & 0xFFFF];
        if (!node.is_leaf) {
            TrieNode node2 = trie[node.child_index + ((ip >> 8) & 0xFF)];
            if (!node2.is_leaf) {
                results[id] = trie[node2.child_index + (ip & 0xFF)].port;
            } else results[id] = node2.port;
        } else results[id] = node.port;
    }
}

// 4. CPU REFERENCE
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

void load_records_from_file(const std::string& filename, std::vector<BGPRecord>& records) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Chyba: Nepodarilo sa otvorit subor " << filename << std::endl;
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
            
            // Logika oktetového priraďovania
            uint8_t first_octet = (ip_val >> 24) & 0xFF;
            int assigned_port = (first_octet % NUM_PORTS) + 1;

            records.push_back({ip_val, mask_val, assigned_port});
            port_stats[assigned_port]++;
        }
    }

    // VÝPIS ŠTATISTIKY 
    std::cout << "\n[STATISTIKA PRIRADENIA PORTOV (" << NUM_PORTS << " portov)]" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    for (int i = 1; i <= NUM_PORTS; i++) {
        std::cout << " Port " << std::setw(2) << i << ": " << std::setw(6) << port_stats[i] << " prefixov";
        if (i % 3 == 0) std::cout << std::endl; else std::cout << " | ";
    }
    std::cout << "----------------------------------------------------" << std::endl;
}

void save_results_to_file(const std::string& filename, const std::vector<unsigned int>& packets, const std::vector<int>& results) {
    std::ofstream outfile(filename);
    outfile << "IP_ADRESA,VYSTUPNY_PORT\n";
    for (size_t i = 0; i < packets.size(); ++i) {
        unsigned int ip = packets[i];
        outfile << ((ip >> 24) & 0xFF) << "." << ((ip >> 16) & 0xFF) << "." << ((ip >> 8) & 0xFF) << "." << (ip & 0xFF) 
                << "," << results[i] << "\n";
    }
}

void run_sweep_test(TrieNode* d_trie, const std::vector<unsigned int>& host_packets) { 
    std::vector<int> test_sizes = { 1024, 8192, 65536, 1048576 };
    
    std::cout << "\n=== SWEEP TEST: GLOBAL MEMORY (RANDOM ACCESS) ===" << std::endl;
    std::cout << "Batch Size | Time (ms) | Throughput (Mpps) | Status (800G)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int size : test_sizes) {
        // POISTKA: Ak chceme testovať viac paketov, než máme v host_packets, preskočíme to
        if (size > host_packets.size()) {
          std::cout << std::setw(10) << size << " | (Exceeds max_batch_size)      | SKIPPED " << std::endl;
          continue;
        }
        unsigned int *d_p; int *d_r;
        cudaMalloc(&d_p, size * sizeof(unsigned int));
        cudaMalloc(&d_r, size * sizeof(int));

        // TERAZ UŽ host_packets POZNÁME, TAK ICH SKOPÍRUJEME
        cudaMemcpy(d_p, host_packets.data(), size * sizeof(unsigned int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        lpm_global_kernel<<<(size + 255)/256, 256>>>(d_p, d_r, d_trie, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms; cudaEventElapsedTime(&ms, start, stop);
        float mpps = (size / (ms / 1000.0f)) / 1000000.0f;
        
        std::cout << std::setw(10) << size << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(9) << ms << " | " 
                  << std::setw(17) << mpps << " | "
                  << (mpps > 1190 ? "OK" : "FAIL") << std::endl;

        cudaFree(d_p); cudaFree(d_r);
    }
}

int main() {
    // A. NAČÍTANIE TESTOVACÍCH PREFIXOV (Pravidlá pre stavbu stromu)
    std::vector<BGPRecord> records;
    // ZMENA: Načítavame z test_prefixes.txt
    load_records_from_file("test_prefixesIPv4.txt", records);
    if (records.empty()) {
        std::cerr << "Chyba: test_prefixes.txt nenajdeny alebo prazdny!" << std::endl;
        return 1;
    }

    std::vector<TrieNode> host_table;
    host_table.reserve(10 * 1024 * 1024); 
    build_real_trie_16_8_8(host_table, records);

    // B. NAČÍTANIE TESTOVACÍCH PAKETOV (Konkrétne IP adresy na kontrolu)
    std::vector<unsigned int> host_packets;
    std::vector<std::string> packet_labels;
    std::ifstream pkt_file("test_packetsIPv4.txt");
    std::string line;

    if (!pkt_file.is_open()) {
        std::cerr << "Chyba: test_packets.txt nenajdeny!" << std::endl;
        return 1;
    }

    while (std::getline(pkt_file, line)) {
        if (line.empty() || line[0] == '#') continue; // Preskoč prázdne a komentáre
        // Odstránime prípadný komentár za IP adresou
        std::string ip_part = line.substr(0, line.find_first_of(" #\t"));
        host_packets.push_back(ip_to_uint(ip_part));
        packet_labels.push_back(ip_part);
    }

    const int BATCH_SIZE = (int)host_packets.size();
    if (BATCH_SIZE == 0) {
        std::cerr << "Chyba: test_packets.txt neobsahuje ziadne IP adresy!" << std::endl;
        return 1;
    }

    // C. GPU ALOKACIA
    TrieNode *d_trie; int *d_results; unsigned int *d_packets;
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode)); 
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));         
    cudaMalloc(&d_packets, BATCH_SIZE * sizeof(unsigned int));

    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packets, host_packets.data(), BATCH_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // D. NASTAVENIE TEXTURE OBJECT (Ponechávame podľa pôvodného kódu)
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_trie;
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    resDesc.res.linear.sizeInBytes = host_table.size() * sizeof(TrieNode);
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t trieTex = 0;
    cudaCreateTextureObject(&trieTex, &resDesc, &texDesc, NULL);

    // E. SPUSTENIE KERNELU (Len jeden blok, keďže máme málo testovacích paketov)
    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;

    lpm_global_kernel<<<blocks, threads>>>(d_packets, d_results, d_trie, BATCH_SIZE);
    cudaDeviceSynchronize();

    // F. ZBER VÝSLEDKOV Z GPU
    std::vector<int> gpu_final_res(BATCH_SIZE);
    cudaMemcpy(gpu_final_res.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // G. POROVNANIE S CPU REFERENCIOU
    std::vector<int> cpu_res(BATCH_SIZE);
    lpm_cpu_reference(host_packets, host_table, cpu_res);

    // H. VERIFIKAČNÝ VÝPIS (Human-Readable Report)
    std::cout << "\n====================================================" << std::endl;
    std::cout << "          IPv4 VERIFICATION REPORT (LPM)            " << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << std::left << std::setw(18) << "Vstupna IP" 
              << " | " << std::setw(10) << "GPU Port" 
              << " | " << std::setw(10) << "CPU Port" 
              << " | " << "Status" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::string gpu_out = (gpu_final_res[i] <= 0) ? "DROPPED" : std::to_string(gpu_final_res[i]);
        std::string cpu_out = (cpu_res[i] <= 0) ? "DROPPED" : std::to_string(cpu_res[i]);

        std::cout << std::left << std::setw(18) << packet_labels[i] 
                  << " | " << std::setw(10) << gpu_out 
                  << " | " << std::setw(10) << cpu_out;
        
        if (gpu_final_res[i] == cpu_res[i]) {
            std::cout << " | [ MATCH ]" << std::endl;
        } else {
            std::cout << " | [ ERROR! ]" << std::endl;
        }
    }
    std::cout << "====================================================" << std::endl;

    // I. ČISTENIE
    cudaDestroyTextureObject(trieTex);
    cudaFree(d_trie); cudaFree(d_results); cudaFree(d_packets);
    return 0;
}