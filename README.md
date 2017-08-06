This is my side project of Udacity course cs344. I try to implement the Blelloch scan without taking advantage of any library. During development, I read the [doc](https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf) carefully. The scan function can accept the array with arbitrary length and primitive type. I tested it on int and size_t, which rough upper bounder are 2^27 and 2^26, respectively. Over that, cudaMalloc() returns "out of memory".

I also implement an inclusive scan base on exclusive scan.

1. mkdir build && cd build
2. cmake -G"Visual Studio 14 Win64" ..  // generate a project for visual studio 14 64bit
3. cmake -G Xcode ..                    // for xcode
4. cmake ..                             // for terminal
