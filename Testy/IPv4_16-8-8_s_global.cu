#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h> 
#include <chrono>         
#include <fstream>        
#include <iomanip>
#include <omp.h>
#include <thread> // Potrebné pre sleep

#include <thrust/device_vector.h> 
#include <thrust/sort.h>          
#include <thrust/execution_policy.h>

// Vlastný alokátor pre Pinned Memory
template <typename T>
struct CudaPinnedAllocator {
    using value_type = T;

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        // Alokujeme Pinned (Page-locked) pamäť
        cudaError_t err = cudaMallocHost((void**)&ptr, n * sizeof(T));
        if (err != cudaSuccess) throw std::bad_alloc();
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        cudaFreeHost(ptr); // Správne uvoľnenie Pinned pamäte
    }
};

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

void log_to_csv(const std::string& filename, const std::string& data) {
    std::ofstream file;
    file.open(filename, std::ios_base::app); 
    if (file.is_open()) {
        file << data << "\n";
        file.close();
    }
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
__global__ void lpm_global_kernel(const unsigned int* __restrict__ packets, int* __restrict__ results, const TrieNode* __restrict__ trie, int batch_size) {
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
template <typename Alloc>
void lpm_cpu_reference(const std::vector<unsigned int, Alloc>& packets, int n_threads, const std::vector<TrieNode>& table, std::vector<int>& results) {
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

template <typename Alloc>
void run_sweep_test(TrieNode* d_trie, const std::vector<unsigned int, Alloc>& host_packets, const std::string& filename) { 
    // --- START WARM-UP PRE SWEEP TEST ---
    const int warm_batch = 1000000;
    unsigned int *d_p_w; int *d_r_w;
    cudaMalloc(&d_p_w, warm_batch * sizeof(unsigned int));
    cudaMalloc(&d_r_w, warm_batch * sizeof(int));
    
    // Tichý beh na prebudenie GPU a PCIe
    lpm_global_kernel<<<(warm_batch + 255) / 256, 256>>>(d_p_w, d_r_w, d_trie, warm_batch);
    cudaDeviceSynchronize();
    
    cudaFree(d_p_w); cudaFree(d_r_w);
    // --- END WARM-UP ---

    std::vector<int> test_sizes = { 1024, 8192, 65536, 524288, 1048576, 5000000, 10000000 };
    
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
                  std::string row = std::to_string(size) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + (mpps >= 1190 ? "OK" : "FAIL");
                  log_to_csv(filename, row);

        cudaFree(d_p); cudaFree(d_r);
    }
    std::cout << "============================================================\n" << std::endl;
}

template <typename Alloc>
void run_cpu_scalability_test(const std::vector<unsigned int, Alloc>& host_packets, const std::vector<TrieNode>& host_table, const std::string& filename) {
    std::vector<int> thread_counts = { 1, 2, 4, 8, 16 };
    const int BATCH_SIZE = (int)host_packets.size();
    std::vector<int> results(BATCH_SIZE);

    std::cout << "\n=== CPU SCALABILITY TEST (AMD Ryzen 7 3700X) ===" << std::endl;
    std::cout << " Threads | Time (ms) | Throughput (Mpps) | Latency (ns) " << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // 1. SILENT WARM-UP (prebudenie všetkých 16 jadier)
    lpm_cpu_reference(host_packets, 16, host_table, results);

    for (int tc : thread_counts) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Volanie tvojej upravenej funkcie s počtom vlákien
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

template <typename Alloc>
void run_gpu_stream_test(TrieNode* d_trie, const std::vector<unsigned int, Alloc>& host_packets, const std::string& filename) {
    std::vector<int> stream_counts = { 2, 3, 4 };
    const int BATCH_SIZE = (int)host_packets.size();
    
    // Alokácia pomocných polí na GPU
    unsigned int *d_p; int *d_r;
    cudaMalloc(&d_p, BATCH_SIZE * sizeof(unsigned int));
    cudaMalloc(&d_r, BATCH_SIZE * sizeof(int));
    std::vector<int, CudaPinnedAllocator<int>> final_res(BATCH_SIZE);

    std::cout << "\n=== GPU STREAM TEST (Pipelining Efficiency) ===" << std::endl;
    std::cout << " Streams | Time (ms) | Throughput (Mpps) | Status (800G) " << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int ns : stream_counts) {
        int chunk_size = BATCH_SIZE / ns;
        
        // Vytvorenie streamov
        std::vector<cudaStream_t> streams(ns);
        for (int i = 0; i < ns; i++) cudaStreamCreate(&streams[i]);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // 1. SILENT WARM-UP (nemeraná časť)
        int warm_threads = 256;
        int warm_blocks = (chunk_size + warm_threads - 1) / warm_threads;
        
        cudaMemcpyAsync(d_p, host_packets.data(), chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[0]);
        // Voláme kernel s explicitne vypočítanými blokmi a vláknami
        lpm_global_kernel<<<warm_blocks, warm_threads, 0, streams[0]>>>(d_p, d_r, d_trie, chunk_size);
        cudaDeviceSynchronize();
        // --- KONIEC WARM-UP ---

        cudaEventRecord(start);
        for (int i = 0; i < ns; i++) {
            int offset = i * chunk_size;
            cudaMemcpyAsync(d_p + offset, host_packets.data() + offset, chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]);
            
            int threads = 256;
            int blocks = (chunk_size + threads - 1) / threads;
            lpm_global_kernel<<<blocks, threads, 0, streams[i]>>>(d_p + offset, d_r + offset, d_trie, chunk_size);
            
            cudaMemcpyAsync(final_res.data() + offset, d_r + offset, chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float ms; cudaEventElapsedTime(&ms, start, stop);
        float mpps = (BATCH_SIZE / (ms / 1000.0f)) / 1000000.0f;

        std::cout << std::setw(8) << ns << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(9) << ms << " | " 
                  << std::setprecision(2) << std::setw(17) << mpps << " | "
                  << (mpps > 1190 ? "OK" : "FAIL") << std::endl;
                  std::string row = std::to_string(ns) + ";" + std::to_string(ms) + ";" + std::to_string(mpps) + ";" + (mpps >= 1190 ? "OK" : "FAIL");
                  log_to_csv(filename, row);

        // Čistenie streamov
        for (int i = 0; i < ns; i++) cudaStreamDestroy(streams[i]);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    
    cudaFree(d_p); cudaFree(d_r);
    std::cout << "========================================================\n" << std::endl;
}

int main() {
    // A. NAČÍTANIE A PRÍPRAVA 
    std::vector<BGPRecord> records;
    load_records_from_file("unique_prefixes.txt", records);
    if (records.empty()) {
        std::cerr << "Chyba: Ziadne zaznamy v unique_prefixes.txt" << std::endl;
        return 1;
    }

    std::vector<TrieNode> host_table;
    host_table.reserve(10 * 1024 * 1024); 
    build_real_trie_16_8_8(host_table, records);

    // Definuj si názvy pre túto sadu testov
    std::string sweep_file = "test2_sweep.csv";
    std::string stream_file = "test2_streams.csv";
    std::string cpu_file = "test2_cpu.csv";
    std::string final_file = "test2_final.csv";

    // Inicializácia súborov (Hlavičky)
    {
        std::ofstream f1(sweep_file); f1 << "Batch Size;Time (ms);Throughput (Mpps);Status\n";
        std::ofstream f2(stream_file); f2 << "Streams;Time (ms);Throughput (Mpps);Status\n";
        std::ofstream f3(cpu_file); f3 << "Threads;Time (ms);Throughput (Mpps);Latency (ns)\n";
        std::ofstream f4(final_file); f4 << "Type;Lat_per_pkt(ns);GPU_Ms;GPU_Mpps;GPU_Eff;CPU_Eff;Zlepsenie;CPU_Ms;Zrychlenie\n";
    }

    // B. GENEROVANIE TESTOVACÍCH PAKETOV 
    const int BATCH_SIZE = 1000000; 
    std::vector<unsigned int, CudaPinnedAllocator<unsigned int>> host_packets(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; i++) {
        host_packets[i] = ((unsigned int)rand() << 16) | (rand() & 0xFFFF);
    }

    // C. GPU ALOKÁCIA
    TrieNode *d_trie; 
    int *d_results; 
    unsigned int *d_packets;

    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode)); 
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));         
    cudaMalloc(&d_packets, BATCH_SIZE * sizeof(unsigned int));

    // D. PRENOS STROMU
    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);

    // E. SPUSTENIE SWEEP TESTU (Pure Kernel Performance)
    // Sweep test meria čistý čas výpočtu na GPU bez započítania PCIe prenosov
    run_sweep_test(d_trie, host_packets, sweep_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // F. SPUSTENIE GPU STREAM TEST
    run_gpu_stream_test(d_trie, host_packets, stream_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // G. Spustenie testu škálovateľnosti CPU
    run_cpu_scalability_test(host_packets, host_table, cpu_file);
    std::this_thread::sleep_for(std::chrono::seconds(10)); 

    // H. NASTAVENIE STREAMS PRE ASYNCHRÓNNY BEH
    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) cudaStreamCreate(&streams[i]);

    int chunk_size = BATCH_SIZE / num_streams;
    std::vector<int, CudaPinnedAllocator<int>> final_res(BATCH_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // I. SPUSTENIE SYSTÉMOVÉHO MERANIA (End-to-End)
    // Meriame celkovú priepustnosť routera vrátane prenosu dát cez PCIe
    
    // --- START WARM-UP ---
    // Definujeme parametre spustenia vopred, aby sme ich mohli použiť pri warm-upe
    int threads_per_block = 256;
    int blocks_per_grid = (chunk_size + threads_per_block - 1) / threads_per_block;

    // 1. Fiktívny prenos (prebudenie PCIe)
    cudaMemcpyAsync(d_packets, host_packets.data(), chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[0]);
    
    // 2. Fiktívny výpočet (prebudenie jadier GPU na Boost frekvenciu)
    lpm_global_kernel<<<blocks_per_grid, threads_per_block, 0, streams[0]>>>(d_packets, d_results, d_trie, chunk_size);
    
    // 3. Synchronizácia - počkáme, kým sa karta "rozhýbe"
    cudaDeviceSynchronize();
    // --- END WARM-UP ---
    
    cudaEventRecord(start); 

    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;

        // 1. Asynchrónny prenos PAKETOV na GPU 
        cudaMemcpyAsync(d_packets + offset, host_packets.data() + offset, 
                        chunk_size * sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]);

        // 2. Spustenie KERNELU (Global Memory prístup)
        int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        lpm_global_kernel<<<blocks, threads, 0, streams[i]>>>(d_packets + offset, d_results + offset, d_trie, chunk_size);

        // 3. Asynchrónny prenos VÝSLEDKOV späť do RAM 
        cudaMemcpyAsync(final_res.data() + offset, d_results + offset, 
                        chunk_size * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop); 
    cudaDeviceSynchronize(); // Počkáme na dokončenie všetkých operácií

    float gpu_ms = 0; 
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // J. POROVNANIE S CPU REFERENCIOU
    auto cpu_s = std::chrono::high_resolution_clock::now();
    std::vector<int> cpu_res(BATCH_SIZE);
    lpm_cpu_reference(host_packets, 16, host_table, cpu_res);
    auto cpu_e = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_e - cpu_s).count();

    // K. ŠTATISTIKY
    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f; 
    float cpu_mpps = (BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f;
    float lat_per_pkt = (gpu_ms * 1000000.0f) / BATCH_SIZE;
    float cpu_lat_per_pkt = (cpu_ms * 1000000.0f) / BATCH_SIZE;
    float gpu_eff = gpu_mpps / 215.0f; 
    float cpu_eff = ((BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f) / 65.0f;

    // VÝPISY NA KONZOLU
    std::cout << "  IPv4 GLOBAL MEMORY - END-TO-END BENCHMARK    " << std::endl;
    std::cout << "\n  Pakety: " << BATCH_SIZE << " | Prefixy: " << records.size() << std::endl;
    std::cout << "  HW: NVIDIA RTX 2070 SUPER | CUDA Streams: " << num_streams << std::endl;
    std::cout << "====================================================" << std::endl;

    std::cout << "\nVYSLEDOK SYSTEMOVEHO MERANIA GPU:" << std::endl;
    std::cout << "  Priemerna systemova latencia: " << std::fixed << std::setprecision(2) << lat_per_pkt << " ns" << std::endl;
    std::cout << "  Cas (Transfer + Kernel):      " << gpu_ms << " ms" << std::endl;
    std::cout << "  Systemova priepustnost:       " << gpu_mpps << " Mpps" << std::endl;

    std::cout << "\nENERGETICKA EFEKTIVITA (Mpk/J):" << std::endl;
    std::cout << "  GPU (System-wide): " << gpu_eff << std::endl; 
    std::cout << "  CPU (System-wide): " << cpu_eff << std::endl;
    std::cout << "  Zlepsenie:         " << gpu_eff / cpu_eff << "x" << std::endl;

    std::cout << "\nPOROVNANIE S CPU:" << std::endl;
    std::cout << "  CPU 16-vlakien: " << cpu_ms << " ms" << std::endl;
    std::cout << "  Zrychlenie:     " << cpu_ms / gpu_ms << "x" << std::endl;
    std::cout << "====================================================" << std::endl;

    // ZÁPIS FINÁLNYCH VÝSLEDKOV
    std::string final_row = "IPv4_Global;" + std::to_string(lat_per_pkt) + ";" + std::to_string(gpu_ms) + ";" + std::to_string(gpu_mpps) + ";" + 
                            std::to_string(gpu_eff) + ";" + std::to_string(cpu_eff) + ";" + std::to_string(gpu_eff / cpu_eff) + ";" + std::to_string(cpu_ms) + ";" + std::to_string(cpu_ms / gpu_ms);
    log_to_csv(final_file, final_row);

    // L. ČISTENIE
    for (int i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_trie); cudaFree(d_results); cudaFree(d_packets);
    return 0;
}