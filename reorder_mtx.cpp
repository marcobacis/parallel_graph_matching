#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

typedef  unsigned long long coord;

int main(int argc, char **argv) {

    if(argc != 2) {
        std::cout << "Usage " << argv[0] << "filename" << std::endl;
        return 1;
    }
    //read from file

    std::string filename = argv[1];

    int height, width, nonzeros;
    coord y,x;

    std::ifstream fin(filename);

    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    fin >> height >> width >> nonzeros;

    std::vector<coord> coords(nonzeros);

    for (int i = 0; i < nonzeros; i++) {
        fin >> y >> x;
        coords[i] = (y-1) * height + (x-1);
    }

    std::stable_sort(coords.begin(), coords.end());

    fin.close();

    std::ofstream fout("sorted.mtx");

    fout << height << " " << width << " " << nonzeros << std::endl;

    for (int i = 0; i < nonzeros; i++) {
        coord y = coords[i] / height + 1;
        coord x = coords[i] % height + 1;

        if (y > height) std::cout << y << std::endl;

        fout << y << " " << x << std::endl;
    }

    fout.close();

}
