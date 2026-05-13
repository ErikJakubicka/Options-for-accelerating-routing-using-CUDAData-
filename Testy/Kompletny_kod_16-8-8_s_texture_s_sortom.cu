#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h> // Základné API pre prácu s NVIDIA GPU
#include <chrono>         // Meranie času na strane CPU
#include <fstream>        // Práca so súbormi (čítanie BGP tabuľky)
#include <iomanip>

// Knižnice Thrust: Slúžia na vysoko-výkonné operácie na GPU (ako STL pre CUDA)
#include <thrust/device_vector.h> // Vektor v pamäti GPU
#include <thrust/sort.h>          // Paralelné radenie priamo na GPU
#include <thrust/execution_policy.h>

// 1. ŠTRUKTÚRA UZLA: alignas(16) zabezpečí, že uzol má presne 16B (128 bitov)
// Umožňuje GPU prečítať celý uzol jedinou inštrukciou 
struct alignas(16) TrieNode {
    int child_index; // Index v poli, kde začínajú deti tohto uzla
    int port;        // Číslo výstupného portu výsledok hľadania
    int is_leaf;     // Príznak (0 alebo 1): Je toto finálny port alebo treba ísť hlbšie?
    int padding;     // "Výplň" na doplnenie do 16B, aby sme udržali zarovnanie pamäte
};

// Pomocná štruktúra pre načítanie riadku z textového súboru
struct BGPRecord {
    uint32_t ip; // IP adresa v binárnom 32-bitovom tvare
    int mask;    // Dĺžka masky (napr. 24 pre /24)
    int port;    // Port priradený danému prefixu
};

// Funkcia prevedie textovú IP na jedno 32-bitové číslo pomocou bitových posunov
uint32_t ip_to_uint(const std::string& ip_str) {
    unsigned int a, b, c, d;
    if (sscanf(ip_str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) return 0;
    // Operátor << posúva bity doľava, | (OR) ich spája do jedného celku
    return (a << 24) | (b << 16) | (c << 8) | d;
}

// 2. BUILDER: Vytvorí v RAM stromovú štruktúru 16-8-8
void build_real_trie_16_8_8(std::vector<TrieNode>& table, std::vector<BGPRecord>& records) {
    table.clear();
    // Prvá úroveň má vždy 2^16 (65536) záznamov
    table.resize(65536, {0, -1, 1, 0}); 

    // KRITICKÝ KROK: Radíme podľa masky od najkratšej po najdlhšiu
    // Dlhšie masky (špecifickejšie) tak neskôr prepíšu tie kratšie (LPM pravidlo)
    std::sort(records.begin(), records.end(), [](const BGPRecord& a, const BGPRecord& b) {
        return a.mask < b.mask;
    });

    for (auto& rec : records) {
        if (rec.mask <= 16) {
            // Level 1: Priamo vyplníme blok v hlavnej tabuľke
            uint32_t startIdx = rec.ip >> 16;
            int numBlocks = 1 << (16 - rec.mask); // Výpočet rozsahu (Prefix Expansion)
            for (int i = 0; i < numBlocks; i++) { 
                table[startIdx + i].port = rec.port; 
                table[startIdx + i].is_leaf = 1; 
            }
        } else if (rec.mask <= 24) {
            // Level 2: Ak narazíme na masku > 16, musíme vytvoriť "pod-tabuľku"
            uint32_t l1 = (rec.ip >> 16) & 0xFFFF;
            if (table[l1].is_leaf) {
                // Ak bola bunka doteraz listom, zmeníme ju na smerník na novú úroveň
                int op = table[l1].port; table[l1].is_leaf = 0;
                table[l1].child_index = (int)table.size(); 
                table.resize(table.size() + 256, {0, op, 1, 0}); // Alokujeme 256 detí
            }
            uint32_t l2_s = (rec.ip >> 8) & 0xFF; // Prostredných 8 bitov IP
            int num = 1 << (24 - rec.mask);
            int base = table[l1].child_index;
            for (int i = 0; i < num; i++) { 
                table[base + l2_s + i].port = rec.port; 
                table[base + l2_s + i].is_leaf = 1; 
            }
        } else {
            // Level 3: Spracovanie najdlhších masiek (do /32)
            // Princíp je rovnaký: hľadáme Level 1 -> Level 2 -> vytvoríme Level 3
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
            uint32_t l3_s = rec.ip & 0xFF; // Posledných 8 bitov IP
            int num = 1 << (32 - rec.mask);
            int b3 = table[b2 + l2].child_index;
            for (int i = 0; i < num; i++) { 
                table[b3 + l3_s + i].port = rec.port; 
                table[b3 + l3_s + i].is_leaf = 1; 
            }
        }
    }
}

// 3. CUDA KERNEL výpočet na GPU
__global__ void lpm_final_kernel(unsigned int* packets, int* results, cudaTextureObject_t trieTex, int batch_size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Unikátne ID vlákna
    if (id < batch_size) {
        unsigned int ip = packets[id];
        
        // 1. Skok: Načítame uzol z Texture Cache (prvých 16 bitov IP)
        // tex1Dfetch<int4> prečíta naraz všetkých 16B dát uzla
        int4 raw = tex1Dfetch<int4>(trieTex, (ip >> 16) & 0xFFFF);
        TrieNode node = *(reinterpret_cast<TrieNode*>(&raw));
        
        if (!node.is_leaf) {
            // 2. Skok: Ideme do druhej úrovne (prostredných 8 bitov)
            raw = tex1Dfetch<int4>(trieTex, node.child_index + ((ip >> 8) & 0xFF));
            TrieNode node2 = *(reinterpret_cast<TrieNode*>(&raw));
            if (!node2.is_leaf) {
                // 3. Skok: Ideme do tretej úrovne (posledných 8 bitov)
                raw = tex1Dfetch<int4>(trieTex, node2.child_index + (ip & 0xFF));
                TrieNode node3 = *(reinterpret_cast<TrieNode*>(&raw));
                results[id] = node3.port;
            } else results[id] = node2.port;
        } else results[id] = node.port;
    }
}

// 4. CPU REFERENCE: Pomalé sekvenčné spracovanie na jednom jadre CPU
void lpm_cpu_reference(const std::vector<unsigned int>& packets, const std::vector<TrieNode>& table, std::vector<int>& results) {
    for (size_t i = 0; i < packets.size(); ++i) {
        unsigned int ip = packets[i];
        TrieNode node = table[(ip >> 16) & 0xFFFF];
        // Logika je rovnaká ako v kerneli, ale beží po jednom pakete v cykle
        if (!node.is_leaf) {
            TrieNode node2 = table[node.child_index + ((ip >> 8) & 0xFF)];
            if (!node2.is_leaf) results[i] = table[node2.child_index + (ip & 0xFF)].port;
            else results[i] = node2.port;
        } else results[i] = node.port;
    }
}

// Funkcia parsuje riadky typu "IP/MASK" a priraďuje port v závislosti od prvého oktetu
void load_records_from_file(const std::string& filename, std::vector<BGPRecord>& records) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Chyba: Nepodarilo sa otvorit subor " << filename << std::endl;
        return;
    }

    std::string line;
    const int NUM_PORTS = 48; // Počet portov podľa požiadavky 
    std::vector<int> port_stats(NUM_PORTS + 1, 0); // Pole na počítanie prefixov na každom porte

    while (std::getline(file, line)) {
        size_t slash = line.find('/');
        if (slash != std::string::npos) {
            uint32_t ip_val = ip_to_uint(line.substr(0, slash));
            int mask_val = std::stoi(line.substr(slash + 1));
            
            // --- LOGIKA OKTETOVÉHO PRIRAĎOVANIA ---
            // Získame prvý oktet (najvýznamnejších 8 bitov)
            uint8_t first_octet = (ip_val >> 24) & 0xFF;
            
            // Výpočet portu: (0-255 % NUM_PORTS) + 1 
            int assigned_port = (first_octet % NUM_PORTS) + 1;

            records.push_back({ip_val, mask_val, assigned_port});
            
            // Započítame do štatistiky
            port_stats[assigned_port]++;
        }
    }

    // Výpis štatistiky hneď po načítaní
    std::cout << "\n[STATISTIKA PRIRADENIA PORTOV (" << NUM_PORTS << " portov)]" << std::endl;
    for (int i = 1; i <= NUM_PORTS; i++) {
        std::cout << " Port " << std::setw(2) << i << ": " << std::setw(6) << port_stats[i] << " prefixov";
        if (i % 3 == 0) std::cout << std::endl; else std::cout << " | ";
    }
    std::cout << std::endl;
}

// Funkcia uloží finálne dvojice (IP adresa, Port) do CSV súboru
void save_results_to_file(const std::string& filename, const std::vector<unsigned int>& packets, const std::vector<int>& results) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) return;
    outfile << "IP_ADRESA,VYSTUPNY_PORT\n";
    for (size_t i = 0; i < packets.size(); ++i) {
        unsigned int ip = packets[i];
        // Rozklad 32-bitového čísla späť na bodkovú notáciu a,b,c,d
        outfile << ((ip >> 24) & 0xFF) << "." << ((ip >> 16) & 0xFF) << "." << ((ip >> 8) & 0xFF) << "." << (ip & 0xFF) 
                << "," << results[i] << "\n";
    }
}

int main() {
    // A. NAČÍTANIE A PRÍPRAVA
    std::vector<BGPRecord> records;
    load_records_from_file("unique_prefixes.txt", records);
    if (records.empty()) return 1;

    std::vector<TrieNode> host_table;
    host_table.reserve(10 * 1024 * 1024); // Rezervácia pamäte pre strom (rýchlosť)
    build_real_trie_16_8_8(host_table, records);

    // B. GENEROVANIE TESTOVACÍCH PAKETOV
    const int BATCH_SIZE = 100000000;
    std::vector<unsigned int> host_packets(BATCH_SIZE);
    host_packets[0] = ip_to_uint("104.153.198.215"); // Manuálne vložená adresa pre verifikáciu
    
    for(int i = 1; i < BATCH_SIZE; i += 10) {
        unsigned int base_ip = ((unsigned int)rand() << 16) | (rand() & 0xFFFF);
        for(int j = 0; j < 10 && (i+j) < BATCH_SIZE; j++) {
            host_packets[i+j] = base_ip + (rand() % 256); // Simulácia tokov (IP blízko seba)
        }
    }

    // C. GPU PAMÄŤ A PRENOS
    TrieNode *d_trie; int *d_results;
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode)); // Alokácia stromu na GPU
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));         // Alokácia poľa pre výsledky
    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);

    // D. OPTIMALIZÁCIA: RADENIE PAKETOV (Thrust)
    auto t_sort_start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<unsigned int> d_packets_vec = host_packets;
    thrust::sort(d_packets_vec.begin(), d_packets_vec.end()); // Zoradenie IP adries priamo na GPU
    auto t_sort_end = std::chrono::high_resolution_clock::now();
    float sort_ms = std::chrono::duration<float, std::milli>(t_sort_end - t_sort_start).count();

    // E. NASTAVENIE TEXTURE OBJECT (Najrýchlejšia cesta k dátam)
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_trie;
    // Formát int4 znamená 4x 32-bit (všetky 4 premenné v TrieNode naraz)
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    resDesc.res.linear.sizeInBytes = host_table.size() * sizeof(TrieNode);
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t trieTex = 0;
    cudaCreateTextureObject(&trieTex, &resDesc, &texDesc, NULL); // Vytvorenie "okuliarov" pre textúru

    // F. PARALELNÉ SPUSTENIE (Streams)
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1); cudaStreamCreate(&stream2); // Vytvorenie dvoch paralelných ciest

    int threads = 256; // Počet vlákien v jednom bloku
    int half = BATCH_SIZE / 2;
    int blocks_half = (half + threads - 1) / threads; // Výpočet počtu blokov

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    unsigned int* d_packets_ptr = thrust::raw_pointer_cast(d_packets_vec.data());
    
    cudaEventRecord(start); // Štart merania času GPU
    // Spúšťame kernel v dvoch streamoch naraz
    lpm_final_kernel<<<blocks_half, threads, 0, stream1>>>(d_packets_ptr, d_results, trieTex, half);
    lpm_final_kernel<<<blocks_half, threads, 0, stream2>>>(d_packets_ptr + half, d_results + half, trieTex, half);
    cudaEventRecord(stop); // Koniec merania času GPU
    
    cudaDeviceSynchronize(); // Počkáme na dokončenie všetkých operácií
    float gpu_ms = 0; cudaEventElapsedTime(&gpu_ms, start, stop);

    // G. ZBER VÝSLEDKOV A ANALÝZA
    std::vector<int> final_res(BATCH_SIZE);
    std::vector<unsigned int> sorted_packets_host(BATCH_SIZE);
    cudaMemcpy(final_res.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_packets_host.data(), d_packets_ptr, BATCH_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //save_results_to_file("routing_results_optimized.csv", sorted_packets_host, final_res);

    // H. POROVNANIE S CPU (Referenčný test)
    auto cpu_s = std::chrono::high_resolution_clock::now();
    std::vector<int> cpu_res(BATCH_SIZE);
    lpm_cpu_reference(host_packets, host_table, cpu_res);
    auto cpu_e = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_e - cpu_s).count();

    // I. VÝPOČET ŠTATISTÍK
    float gpu_mpps = (BATCH_SIZE / (gpu_ms / 1000.0f)) / 1000000.0f; // Milióny paketov za sekundu
    float gpu_eff = gpu_mpps / 215.0f; // Energetická efektivita (odhadovaná spotreba RTX 2070S)
    float cpu_eff = ((BATCH_SIZE / (cpu_ms / 1000.0f)) / 1000000.0f) / 65.0f; // Odhad pre CPU

    // VÝPISY NA KONZOLU
    std::cout << "====================================================" << std::endl;
    std::cout << "TEXTURE CACHE + THRUST SORT (COALESCED)" << std::endl;
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

    // J. ČISTENIE PAMÄTE
    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
    cudaDestroyTextureObject(trieTex);
    cudaFree(d_trie); cudaFree(d_results);
    return 0;
}