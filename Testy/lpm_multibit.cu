#include <iostream>
#include <vector>
#include <cuda_runtime.h>

struct TrieNode {
    int child_index; 
    int port;        
    bool is_leaf;    
};

// --- GPU KERNEL: 2-úrovňové LPM vyhľadávanie ---
__global__ void lpm_trie_multi_level_kernel(unsigned int* packets, int* results, TrieNode* trie, int batch_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        unsigned int ip = packets[id];
        
        // 1. ÚROVEŇ (Spracujeme bity 31-28)
        int current_node_idx = 0; // Začíname v koreni
        unsigned int stride1 = (ip >> 28) & 0x0F; 
        int next_node_idx = trie[current_node_idx].child_index + stride1;
        
        // 2. ÚROVEŇ (Spracujeme bity 27-24)
        // Ak uzol v 1. úrovni nie je list, pokračujeme hlbšie
        if (!trie[next_node_idx].is_leaf) {
            unsigned int stride2 = (ip >> 24) & 0x0F;
            next_node_idx = trie[next_node_idx].child_index + stride2;
        }
        
        results[id] = trie[next_node_idx].port;
    }
}

void build_multi_level_trie(std::vector<TrieNode>& table) {
    // 0. Koreňový uzol
    TrieNode root = {1, -1, false}; 
    table.push_back(root);

    // 1. Úroveň: Vytvoríme 16 detí (pre prvý 4-bit stride)
    for(int i = 0; i < 16; i++) {
        TrieNode node;
        if (i == 5) { 
            // Uzol pre prefix 0x5... nebude list, ale odkáže na 2. úroveň
            node.child_index = 17; // Deti 2. úrovne začnú za prvými 16 deťmi
            node.port = -1;
            node.is_leaf = false;
        } else {
            node.child_index = 0;
            node.port = 100 + i; // Porty 100-115
            node.is_leaf = true;
        }
        table.push_back(node);
    }

    // 2. Úroveň: Pridáme 16 detí pre uzol s indexom 5 (prefix 0x5...)
    for(int j = 0; j < 16; j++) {
        TrieNode node = {0, 200 + j, true}; // Porty 200-215
        table.push_back(node);
    }
}

int main() {
    // 1. Zvýšime dávku na 1 milión paketov
    const int BATCH_SIZE = 1000000; 
    std::vector<TrieNode> host_table;
    build_multi_level_trie(host_table);

    // 2. Vygenerujeme náhodné IP adresy (zapotíme procesor)
    std::vector<unsigned int> host_packets(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++) {
        // Generujeme IP adresy, kde časť bude začínať 0x5... aby sme trafili 2. úroveň
        host_packets[i] = (rand() % 2 == 0) ? 0x50000000 | (rand() % 0x0FFFFFFF) : rand();
    }
    std::vector<int> host_results(BATCH_SIZE);

    // Alokácia a prenos (ako predtým, len s BATCH_SIZE)
    TrieNode *d_trie; unsigned int *d_packets; int *d_results;
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode));
    cudaMalloc(&d_packets, BATCH_SIZE * sizeof(unsigned int));
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));

    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packets, host_packets.data(), BATCH_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // --- MERANIE ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Výpočet počtu blokov: (Celkový počet paketov / 1024 vlákien na blok)
    // RTX 2070 SUPER má limit 1024 vlákien na jeden blok
    int threadsPerBlock = 1024;
    int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    lpm_trie_multi_level_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_packets, d_results, d_trie, BATCH_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Získame výsledky späť pre kontrolu (stačí prvých pár)
    cudaMemcpy(host_results.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "--- ZATAZOVY TEST: 1 MILION PAKETOV ---" << std::endl;
    std::cout << "Pocet blokov: " << blocksPerGrid << ", Vlakien na blok: " << threadsPerBlock << std::endl;
    std::cout << "Cas na GPU: " << milliseconds << " ms" << std::endl;

    if (milliseconds > 0) {
        float mpaks = (BATCH_SIZE / (milliseconds / 1000.0f)) / 1000000.0f;
        std::cout << "VYSLEDNA PRIEPUSTNOST: " << mpaks << " Mpak/s" << std::endl;
    }

    // Upratovanie...
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_trie); cudaFree(d_packets); cudaFree(d_results);
    return 0;
}