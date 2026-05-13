#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "Chyba: Ziadna CUDA grafika sa nenasla!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 je tvoja RTX 2070 SUPER

    std::cout << "Zariadenie: " << prop.name << std::endl;
    std::cout << "Pocet multiprocesorov (SM): " << prop.multiProcessorCount << std::endl;
    std::cout << "Max. vlakien na blok: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Globalna pamat: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    return 0;
}