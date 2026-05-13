#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>


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

// --- BUILDER 16-8-8 ---
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
            for (int i = 0; i < numBlocks; i++) { table[startIdx + i].port = rec.port; table[startIdx + i].is_leaf = 1; }
        } else if (rec.mask <= 24) {
            uint32_t l1 = (rec.ip >> 16) & 0xFFFF;
            if (table[l1].is_leaf) {
                int op = table[l1].port; table[l1].is_leaf = 0;
                table[l1].child_index = (int)table.size(); table.resize(table.size() + 256, {0, op, 1, 0});
            }
            uint32_t l2_s = (rec.ip >> 8) & 0xFF;
            int num = 1 << (24 - rec.mask);
            int base = table[l1].child_index;
            for (int i = 0; i < num; i++) { table[base + l2_s + i].port = rec.port; table[base + l2_s + i].is_leaf = 1; }
        } else {
            uint32_t l1 = (rec.ip >> 16) & 0xFFFF;
            if (table[l1].is_leaf) {
                int op = table[l1].port; table[l1].is_leaf = 0;
                table[l1].child_index = (int)table.size(); table.resize(table.size() + 256, {0, op, 1, 0});
            }
            uint32_t l2 = (rec.ip >> 8) & 0xFF;
            int b2 = table[l1].child_index;
            if (table[b2 + l2].is_leaf) {
                int op = table[b2 + l2].port; table[b2 + l2].is_leaf = 0;
                table[b2 + l2].child_index = (int)table.size(); table.resize(table.size() + 256, {0, op, 1, 0});
            }
            uint32_t l3_s = rec.ip & 0xFF;
            int num = 1 << (32 - rec.mask);
            int b3 = table[b2 + l2].child_index;
            for (int i = 0; i < num; i++) { table[b3 + l3_s + i].port = rec.port; table[b3 + l3_s + i].is_leaf = 1; }
        }
    }
}

// --- CUDA KERNEL ---
__global__ void lpm_final_kernel(unsigned int* packets, int* results, cudaTextureObject_t trieTex, int batch_size) {
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

// --- CPU REFERENCE ---
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

void update_prefix_on_gpu(TrieNode* d_table, const std::vector<TrieNode>& h_table, uint32_t ip, int mask, int new_port) {
    if (mask <= 16) {
        uint32_t startIdx = ip >> 16;
        int numBlocks = 1 << (16 - mask);
        for (int i = 0; i < numBlocks; i++) {
            size_t offset = (startIdx + i) * sizeof(TrieNode) + offsetof(TrieNode, port);
            cudaMemcpy((char*)d_table + offset, &new_port, sizeof(int), cudaMemcpyHostToDevice);
        }
    }
}

void load_records_from_file(const std::string& filename, std::vector<BGPRecord>& records) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    std::string line;
    int p_cnt = 1;
    while (std::getline(file, line)) {
        size_t slash = line.find('/');
        if (slash != std::string::npos) {
            records.push_back({ip_to_uint(line.substr(0, slash)), std::stoi(line.substr(slash + 1)), (p_cnt % 1000) + 1});
            p_cnt++;
        }
    }
}

void save_results_to_file(const std::string& filename, const std::vector<unsigned int>& packets, const std::vector<int>& results) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Chyba pri vytvarani suboru s vysledkami!" << std::endl;
        return;
    }

    outfile << "IP_ADRESA,VYSTUPNY_PORT" << std::endl; // Hlavička pre CSV formát

    for (size_t i = 0; i < packets.size(); ++i) {
        unsigned int ip = packets[i]; 
        // Prevod na oktety
        unsigned char o1 = (ip >> 24) & 0xFF;
        unsigned char o2 = (ip >> 16) & 0xFF;
        unsigned char o3 = (ip >> 8) & 0xFF;
        unsigned char o4 = ip & 0xFF;

        outfile << (int)o1 << "." << (int)o2 << "." << (int)o3 << "." << (int)o4 
                << "," << results[i] << "\n";
    }

    outfile.close();
    std::cout << "Vysledky boli uspesne ulozene do suboru: " << filename << std::endl;
}

int main() {
    std::vector<BGPRecord> records;
    load_records_from_file("unique_prefixes.txt", records);
    if (records.empty()) return 1;

    std::vector<TrieNode> host_table;
    host_table.reserve(10 * 1024 * 1024);
    build_real_trie_16_8_8(host_table, records);

    const int BATCH_SIZE = 1000000;
    std::vector<unsigned int> host_packets(BATCH_SIZE);
    host_packets[0] = ip_to_uint("147.175.10.10");
    for(int i = 1; i < BATCH_SIZE; i++) host_packets[i] = ((unsigned int)rand() << 16) | (rand() & 0xFFFF);

    TrieNode *d_trie; unsigned int *d_packets; int *d_results;
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode));
    cudaMalloc(&d_packets, BATCH_SIZE * sizeof(unsigned int));
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));
    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packets, host_packets.data(), BATCH_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_trie;
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    resDesc.res.linear.sizeInBytes = host_table.size() * sizeof(TrieNode);
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t trieTex = 0;
    cudaCreateTextureObject(&trieTex, &resDesc, &texDesc, NULL);

    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    lpm_final_kernel<<<blocks, threads>>>(d_packets, d_results, trieTex, BATCH_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0; cudaEventElapsedTime(&gpu_ms, start, stop);

    std::vector<int> gpu_res(BATCH_SIZE);
    cudaMemcpy(gpu_res.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> res_after(BATCH_SIZE);
    cudaMemcpy(res_after.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // ULOŽENIE DO SÚBORU
    save_results_to_file("routing_results.csv", host_packets, res_after);

    auto cpu_s = std::chrono::high_resolution_clock::now();
    std::vector<int> cpu_res(BATCH_SIZE);
    lpm_cpu_reference(host_packets, host_table, cpu_res);
    auto cpu_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_ms = cpu_e - cpu_s;

    // --- ENERGETICKÁ EFEKTIVITA ---
    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f;
    float gpu_tdp = 215.0f; // RTX 2070 Super TDP in Watts
    float cpu_tdp = 65.0f;  // Priemerné CPU TDP
    float gpu_eff = gpu_mpps / gpu_tdp; // Mpk / Joule
    float cpu_eff = ((BATCH_SIZE / (cpu_ms.count() / 1000.0f)) / 1000000.0f) / cpu_tdp;

    std::cout << "====================================================" << std::endl;
    std::cout << "NASTAVENIE TESTU:" << std::endl;
    std::cout << "  Pocet paketov: " << BATCH_SIZE << " | Prefixov: " << records.size() << std::endl;
    std::cout << "  GPU: NVIDIA RTX 2070 SUPER (TDP: 215W)" << std::endl;
    std::cout << "====================================================" << std::endl;

    std::cout << "\nVYSLEDOK MERANIA:" << std::endl;
    std::cout << "  Cas na GPU:    " << gpu_ms << " ms" << std::endl;
    std::cout << "  Priepustnost:  " << gpu_mpps << " Mpps" << std::endl;

    std::cout << "\nENERGETICKA EFEKTIVITA (Mpk/J):" << std::endl;
    std::cout << "  GPU Efektivita: " << gpu_eff << " Mpk/J" << std::endl;
    std::cout << "  CPU Efektivita: " << cpu_eff << " Mpk/J" << std::endl;
    std::cout << "  Zlepsenie:      " << gpu_eff / cpu_eff << "x efektivnejsie" << std::endl;

    std::cout << "\nPOROVNANIE CPU VS GPU:" << std::endl;
    std::cout << "  Cas na CPU:    " << cpu_ms.count() << " ms" << std::endl;
    std::cout << "  Zrychlenie:    " << cpu_ms.count() / gpu_ms << "x" << std::endl;
    std::cout << "====================================================" << std::endl;

    cudaDestroyTextureObject(trieTex);
    cudaFree(d_trie); cudaFree(d_packets); cudaFree(d_results);
    return 0;
}