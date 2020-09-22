#ifndef _STOPWATCH_H_
#define _STOPWATCH_H_

#include <iostream>
#if defined(__linux__)
#   include <sys/times.h>
#elif defined(WIN32) || defined(_WIN64)
#   include <windows.h>
#endif

class Stopwatch {
public:
    Stopwatch(const char* taskname) : taskname_(taskname), start_(now()), last_tick_(now()) {
        //std::cout << taskname_ << "..." << std::endl;
    }

    ~Stopwatch() {
        double elapsed = now() - start_;
        std::cout << taskname_ << ": " << elapsed << "s" << std::endl;
    }

    void tick(const char* tickname="tick") {
        double elapsed = now() - start_;
        std::cout << taskname_ << "==>" << tickname << " :  delta = " << now() - last_tick_ << "s    sum= " << elapsed << std::endl;
        last_tick_ = now();
    }

    static double now() {
#if defined(__linux__)
        tms now_tms;
        return double(times(&now_tms)) / 100.0;
#elif defined(WIN32) || defined(_WIN64)
        return double(GetTickCount()) / 1000.0;
#else
        return 0.0;
#endif
    }

private:
    const char* taskname_;
    double start_;
    double last_tick_;
};

#endif // _STOPWATCH_H_

