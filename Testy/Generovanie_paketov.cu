#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// 1. Toto je kód, ktorý beží priamo na GPU jadrách
__global__ void lpm_lookup_kernel(unsigned int* packets, int* results, int batch_size) {
    // Vypočítame unikátne ID vlákna
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < batch_size) {
        // Simulácia Longest Prefix Match (LPM)
        // V reálnej verzii tu bude prechod cez Multibit Trie (CSF/cuTS)
        unsigned int ip = packets[id];
        
        // Jednoduchá podmienka pre ukážku: ak je IP párna, smeruj na port 1, inak na port 2
        if (ip % 2 == 0) {
            results[id] = 1; 
        } else {
            results[id] = 2;
        }
    }
}

int main() {
    const int BATCH_SIZE = 1024;
    size_t size_p = BATCH_SIZE * sizeof(unsigned int);
    size_t size_r = BATCH_SIZE * sizeof(int);

    // Príprava dát na CPU
    std::vector<unsigned int> host_packets(BATCH_SIZE);
    std::vector<int> host_results(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++) host_packets[i] = i; 

    // Alokácia na GPU
    unsigned int *d_packets;
    int *d_results;
    cudaMalloc(&d_packets, size_p);
    cudaMalloc(&d_results, size_r);

    // Kopírovanie do GPU
    cudaMemcpy(d_packets, host_packets.data(), size_p, cudaMemcpyHostToDevice);

    // 2. SPUSTENIE KERNELU (1024 vlákien na RTX 2070 SUPER)
    // 1 blok po 1024 vláknach (max. limit tvojej karty)
    lpm_lookup_kernel<<<1, 1024>>>(d_packets, d_results, BATCH_SIZE);

    // Kopírovanie výsledkov späť do CPU
    cudaMemcpy(host_results.data(), d_results, size_r, cudaMemcpyDeviceToHost);

    // Výpis prvých 10 výsledkov
    std::cout << "Vysledky smerovania (prvych 10 paketov):" << std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout << "Paket " << i << " (IP: " << host_packets[i] << ") -> Odoslane na Port: " << host_results[i] << std::endl;
    }

    cudaFree(d_packets);
    cudaFree(d_results);
    return 0;
}