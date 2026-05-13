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

void run_sweep_test(TrieNode* d_trie, cudaTextureObject_t trieTex, const std::vector<unsigned int>& host_packets) {
    // Skúšame rôzne veľkosti balíkov
    std::vector<int> test_sizes = { 1024, 8192, 65536, 524288, 1048576 };
    
    std::cout << "\n=== SWEEP TEST: TEXTURE CACHE (RANDOM ACCESS) ===" << std::endl;
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

        // Kopírujeme reálne dáta pre férový test cache
        cudaMemcpy(d_p, host_packets.data(), size * sizeof(unsigned int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        // Voláme tvoj lpm_final_kernel (ktorý používa textúry)
        lpm_final_kernel<<<(size + 255)/256, 256>>>(d_p, d_r, trieTex, size);
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
    std::cout << "============================================================\n" << std::endl;
}


int main() {
    // A. NAČÍTANIE TESTOVACÍCH PREFIXOV
    std::vector<BGPRecord> records;
    load_records_from_file("test_prefixesIPv4.txt", records);
    if (records.empty()) {
        std::cerr << "Chyba: test_prefixes.txt nenajdeny alebo prazdny!" << std::endl;
        return 1;
    }

    std::vector<TrieNode> host_table;
    host_table.reserve(1024 * 1024); 
    build_real_trie_16_8_8(host_table, records);

    // B. NAČÍTANIE TESTOVACÍCH PAKETOV (Namiesto rand())
    std::vector<unsigned int> host_packets;
    std::vector<std::string> packet_labels;
    std::ifstream pkt_file("test_packetsIPv4.txt");
    std::string line;

    while (std::getline(pkt_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        // Odstránime prípadný komentár za IP adresou
        std::string ip_part = line.substr(0, line.find_first_of(" #\t"));
        host_packets.push_back(ip_to_uint(ip_part));
        packet_labels.push_back(ip_part);
    }

    const int BATCH_SIZE = (int)host_packets.size();
    if (BATCH_SIZE == 0) {
        std::cerr << "Chyba: test_packets.txt je prazdny!" << std::endl;
        return 1;
    }

    // C. GPU ALOKÁCIA
    TrieNode *d_trie; 
    int *d_results;
    unsigned int *d_packets_raw;

    // Opravený preklep z tvojho pôvodného kódu (chýbalo d_trie)
    cudaMalloc(&d_trie, host_table.size() * sizeof(TrieNode));
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_packets_raw, BATCH_SIZE * sizeof(unsigned int));

    cudaMemcpy(d_trie, host_table.data(), host_table.size() * sizeof(TrieNode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packets_raw, host_packets.data(), BATCH_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // D. NASTAVENIE TEXTURE OBJECT
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_trie;
    resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
    resDesc.res.linear.sizeInBytes = host_table.size() * sizeof(TrieNode);
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t trieTex = 0;
    cudaCreateTextureObject(&trieTex, &resDesc, &texDesc, NULL);

    // E. SPUSTENIE VÝPOČTU (Pre verifikáciu stačí 1 blok)
    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;
    lpm_final_kernel<<<blocks, threads>>>(d_packets_raw, d_results, trieTex, BATCH_SIZE);
    cudaDeviceSynchronize();

    // F. ZBER VÝSLEDKOV A CPU REFERENCIA
    std::vector<int> final_res(BATCH_SIZE);
    cudaMemcpy(final_res.data(), d_results, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> cpu_res(BATCH_SIZE);
    lpm_cpu_reference(host_packets, host_table, cpu_res);

    // G. VERIFIKAČNÝ VÝPIS (Report pre diplomovku)
    std::cout << "\n====================================================" << std::endl;
    std::cout << "          IPv4 TEXTURE CACHE VERIFICATION           " << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << std::left << std::setw(18) << "Vstupna IP" 
              << " | " << std::setw(10) << "GPU Port" 
              << " | " << std::setw(10) << "CPU Port" 
              << " | " << "Status" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    for (int i = 0; i < BATCH_SIZE; i++) {
        std::string gpu_out = (final_res[i] <= 0) ? "DROPPED" : std::to_string(final_res[i]);
        std::string cpu_out = (cpu_res[i] <= 0) ? "DROPPED" : std::to_string(cpu_res[i]);

        std::cout << std::left << std::setw(18) << packet_labels[i] 
                  << " | " << std::setw(10) << gpu_out 
                  << " | " << std::setw(10) << cpu_out;
        
        if (final_res[i] == cpu_res[i]) {
            std::cout << " | [ MATCH ]" << std::endl;
        } else {
            std::cout << " | [ ERROR! ]" << std::endl;
        }
    }
    std::cout << "====================================================" << std::endl;

    // J. ČISTENIE
    cudaDestroyTextureObject(trieTex);
    cudaFree(d_trie); cudaFree(d_results); cudaFree(d_packets_raw);
    return 0;
}