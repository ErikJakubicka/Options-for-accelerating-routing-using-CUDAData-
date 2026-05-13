#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

// 1. Dôležitá zmena: alignas(16) zabezpečí, že uzol sedí do Texture Cache
struct alignas(16) TrieNode {
    int child_index; 
    int port;        
    int is_leaf;    
    int padding; // Doplníme do 16 bajtov pre hardvérovú kompatibilitu
};

// --- FINÁLNY OPTIMALIZOVANÝ KERNEL ---
__global__ void lpm_texture_kernel(unsigned int* packets, int* results, cudaTextureObject_t trieTex, int batch_size) {
    __shared__ TrieNode shared_root_level[17];

    int tid = threadIdx.x;
    if (tid < 17) {
        // Načítanie do Shared Memory cez Texture Cache
        int4 raw_data = tex1Dfetch<int4>(trieTex, tid);
        shared_root_level[tid] = *(reinterpret_cast<TrieNode*>(&raw_data));
    }
    __syncthreads();

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < batch_size) {
        unsigned int ip = packets[id];
        unsigned int stride1 = (ip >> 28) & 0x0F;
        
        int next_node_idx = shared_root_level[0].child_index + stride1;
        TrieNode node = shared_root_level[next_node_idx];

        if (!node.is_leaf) {
            unsigned int stride2 = (ip >> 24) & 0x0F;
            int deep_idx = node.child_index + stride2;
            
            // FINÁLNY FIX: Načítanie cez int4 pre kompatibilitu s tex1Dfetch
            int4 raw_deep = tex1Dfetch<int4>(trieTex, deep_idx);
            TrieNode deep_node = *(reinterpret_cast<TrieNode*>(&raw_deep));
            
            results[id] = deep_node.port;
        } else {
            results[id] = node.port;
        }
    }
}

// Funkcia na vybudovanie masívnej tabuľky (ako predtým)
void build_massive_trie(std::vector<TrieNode>& table, int ballast_nodes) {
    table.clear(); // Vyčistíme pre istotu
    table.push_back({ 1, -1, 0, 0 }); // Koreň

    for (int i = 0; i < 16; i++) {
        table.push_back({ 17 + (i * 16), -1, 0, 0 });
    }

    for (int j = 0; j < 256; j++) {
        table.push_back({ 0, 300 + j, 1, 0 });
    }

    // Tu meníme veľkosť tabuľky podľa tvojho zadania
    for (int k = 0; k < ballast_nodes; k++) {
        table.push_back({ 0, -1, 1, 0 });
    }
}

// Jednoduchá CPU verzia LPM pre porovnanie
void lpm_cpu_reference(const std::vector<unsigned int>& packets, const std::vector<TrieNode>& trie, std::vector<int>& results) {
    for (size_t i = 0; i < packets.size(); ++i) {
        unsigned int ip = packets[i];
        unsigned int stride1 = (ip >> 28) & 0x0F;
        
        int next_node_idx = trie[0].child_index + stride1;
        TrieNode node = trie[next_node_idx];

        if (!node.is_leaf) {
            unsigned int stride2 = (ip >> 24) & 0x0F;
            int deep_idx = node.child_index + stride2;
            results[i] = trie[deep_idx].port;
        } else {
            results[i] = node.port;
        }
    }
}

int main() {
    const int BATCH_SIZE = 1000000; 
    int nodes_to_add = 50273;       
    std::vector<TrieNode> host_table;
    build_massive_trie(host_table, nodes_to_add);

    std::vector<unsigned int> host_packets(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++) {
        if (rand() % 2 == 0) {
            // 50 % paketov je z testovacieho rozsahu 0x50... (80.x.x.x)
            // Použitie bitového posun, aby sme vyplnili celé 32-bitové číslo
            host_packets[i] = 0x50000000 | ((rand() << 16) | (rand() & 0xFFFF));
        } else {
            // 50 % paketov bude vyzerať ako z univerzitnej siete (147.175.x.x)
            // HEX 0x93AF = 147.175
            host_packets[i] = 0x93AF0000 | (rand() & 0xFFFF); 
        }
    }
    std::vector<int> host_results(BATCH_SIZE);

    TrieNode *d_trie; unsigned int *d_packets; int *d_results;
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode));
    cudaMalloc(&d_packets, BATCH_SIZE * sizeof(unsigned int));
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));

    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packets, host_packets.data(), BATCH_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // 3. Oprava popisu zdroja pre textúru
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_trie;
    // Explicitne povieme, že ide o 4x32-bit (int4)
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    resDesc.res.linear.sizeInBytes = host_table.size() * sizeof(TrieNode);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t trieTex = 0;
    cudaCreateTextureObject(&trieTex, &resDesc, &texDesc, NULL);
    
    // --- MERANIE ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    lpm_texture_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_packets, d_results, trieTex, BATCH_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_results.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // --- PREHĽADNÝ VÝPIS PARAMETROV ---
    std::cout << "====================================================" << std::endl;
    std::cout << "NASTAVENIE TESTU:" << std::endl;
    std::cout << "BATCH SIZE (Pocet paketov):  " << BATCH_SIZE << std::endl;
    std::cout << "POCET UZLOV V TABULKE:       " << host_table.size() << std::endl;
    std::cout << "HARDVER:                     NVIDIA RTX 2070 SUPER" << std::endl; //
    std::cout << "====================================================" << std::endl;

    // Uprav aj finálny výpis, aby bol stručnejší:
    std::cout << "\n--- VYSLEDOK MERANIA ---" << std::endl;
    std::cout << "Cas na GPU:    " << milliseconds << " ms" << std::endl;
    std::cout << "Priepustnost:  " << (BATCH_SIZE / (milliseconds / 1000.0f)) / 1000000.0f << " Mpak/s" << std::endl;
    std::cout << "====================================================" << std::endl;

    // Upratovanie
    cudaDestroyTextureObject(trieTex);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_trie); cudaFree(d_packets); cudaFree(d_results);

    // --- MERANIE NA CPU PRE POROVNANIE ---
    std::vector<int> cpu_results(BATCH_SIZE);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    lpm_cpu_reference(host_packets, host_table, cpu_results);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_ms = end_cpu - start_cpu;

    std::cout << "\n--- POROVNANIE CPU VS GPU ---" << std::endl;
    std::cout << "Cas na CPU:    " << cpu_ms.count() << " ms" << std::endl;
    std::cout << "Zrychlenie:    " << cpu_ms.count() / milliseconds << "x" << std::endl;
    std::cout << "====================================================" << std::endl;

    // --- TEST SPRÁVNOSTI (VALIDÁCIA) ---
    bool match = true;
    for(int i = 0; i < BATCH_SIZE; i++) { 
        if(host_results[i] != cpu_results[i]) {
            std::cout << "!!! CHYBA: Vysledky sa nezhoduju na indexe " << i << "!" << std::endl;
            std::cout << "GPU: " << host_results[i] << " vs CPU: " << cpu_results[i] << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "VALIDACIA USPESNA: GPU a CPU vysledky su identicke." << std::endl;
        std::cout << "====================================================" << std::endl;
    }

    // --- VIZUALIZÁCIA VZORKY DÁT ---
    std::cout << "\nVZORKA SPRACOVANIA (Prvych " << BATCH_SIZE << " paketov):" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << "ID\tHEX Adresa\tIPv4 Adresa\t\tCielovy Port" << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;

    int limit_zobrazenia = std::min(BATCH_SIZE, 15); 

    for (int i = 0; i < limit_zobrazenia; i++) {
        unsigned int ip = host_packets[i];
        
        // Extrakcia oktetov pre IPv4 format
        unsigned char o1 = (ip >> 24) & 0xFF;
        unsigned char o2 = (ip >> 16) & 0xFF;
        unsigned char o3 = (ip >> 8) & 0xFF;
        unsigned char o4 = ip & 0xFF;

        printf("%d\t0x%08X\t%u.%u.%u.%u\t\tPort: %d\n", 
               i, ip, o1, o2, o3, o4, host_results[i]);
    }
    std::cout << "----------------------------------------------------------------------" << std::endl;

    // --- VIZUALIZÁCIA STRUKTURY STROMU (Korenova uroven) ---
    std::cout << "\nNAHLAD DO SMEROVACEJ TABULKY (Shared Memory Level):" << std::endl;
    std::cout << "Index\tChild_Idx\tPort\tLeaf?" << std::endl;
    int nodes_to_show = std::min((int)host_table.size(), 17); 

    for (int i = 0; i < nodes_to_show; i++) {
        printf("[%d]\t%d\t\t%d\t%s\n", 
            i, host_table[i].child_index, host_table[i].port, 
            host_table[i].is_leaf ? "YES" : "NO");
    }
    std::cout << "====================================================" << std::endl;
    return 0;
}