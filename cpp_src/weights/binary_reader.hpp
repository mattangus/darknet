#pragma once

#include <vector>
#include <fstream>

namespace darknet
{
namespace weights
{
    class BinaryReader
    {
    private:
    public:
        std::ifstream file;
        BinaryReader(std::string path) : file(path, std::ios::in | std::ios::binary) {

        }
        ~BinaryReader() {}

        bool is_open() { return file.is_open(); }
        bool eof() { return file.eof(); }

        template<typename T>
        T read() {
            T v;
            file.read((char*)&v, sizeof(T));
            return v;
        }

        template<typename T>
        std::vector<T> readN(int n)
        {
            std::vector<T> ret(n); // assumes default constructor
            file.read((char*)&ret[0], n*sizeof(T));
            return ret;
        }

        int bytesLeft()
        {
            //TODO: Reset location
            auto start = file.tellg();
            file.seekg( 0, std::ios::end );
            auto fsize = file.tellg() - start;
            
            return fsize;
        }
    };
} // namespace weights
} // namespace darknet
