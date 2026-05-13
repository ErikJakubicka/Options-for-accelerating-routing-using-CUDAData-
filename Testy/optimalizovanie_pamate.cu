#include <iostream>
#include <vector>
#include <cuda_runtime.h>

struct TrieNode {
    int child_index; 
    int port;        
    bool is_leaf;    
};

// --- OPTIMALIZOVANÝ KERNEL ---
__global__ void lpm_optimized_kernel(unsigned int* packets, int* results, TrieNode* trie, int batch_size) {
    // 1. Definícia Shared Memory pre 17 uzlov (koreň + prvá úroveň)
    __shared__ TrieNode shared_root_level[17];

    // 2. Len prvé vlákna v bloku skopírujú dáta z VRAM do Shared Memory
    int tid = threadIdx.x;
    if (tid < 17) {
        shared_root_level[tid] = trie[tid];
    }
    __syncthreads(); // Počkáme, kým je Shared pamäť naplnená

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        unsigned int ip = packets[id];
        
        // 1. ÚROVEŇ: Čítame z bleskurýchlej SHARED pamäte
        int current_node_idx = 0; 
        unsigned int stride1 = (ip >> 28) & 0x0F; 
        int next_node_idx = shared_root_level[current_node_idx].child_index + stride1;
        TrieNode current_node = shared_root_level[next_node_idx];

        // 2. ÚROVEŇ: Ak ideme hlbšie, čítame cez TEXTURE CACHE (__ldg)
        if (!current_node.is_leaf) {
            unsigned int stride2 = (ip >> 24) & 0x0F;
            int deep_node_idx = current_node.child_index + stride2;
            
            // __ldg povie GPU: "Použi Texture/L1 cache pre toto čítanie"
            results[id] = __ldg(&trie[deep_node_idx].port); 
        } else {
            results[id] = current_node.port;
        }
    }
}

void build_massive_trie(std::vector<TrieNode>& table) {
    // 0. Koreň
    table.push_back({1, -1, false});

    // 1. Úroveň (16 detí)
    for(int i = 0; i < 16; i++) {
        // Každé dieťa teraz odkáže na vlastnú sekciu v 2. úrovni
        table.push_back({17 + (i * 16), -1, false});
    }

    // 2. Úroveň (16 * 16 = 256 uzlov)
    // Toto už začne vytláčať dáta z L1 cache pri masívnom prístupe
    for(int j = 0; j < 256; j++) {
        table.push_back({0, 300 + j, true});
    }
    
    // Voliteľne: môžeme pridať tisíce "balastných" uzlov, 
    // aby sme tabuľku roztiahli v pamäti na niekoľko MB.
    for(int k = 0; k < 50000; k++) {
        table.push_back({0, -1, true});
    }
}

int main() {
    // 1. Zvýšime dávku na 1 milión paketov
    const int BATCH_SIZE = 1000000; 
    std::vector<TrieNode> host_table;
    build_massive_trie(host_table);

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

    lpm_optimized_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_packets, d_results, d_trie, BATCH_SIZE);

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