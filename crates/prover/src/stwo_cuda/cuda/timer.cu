#include "timer.cuh"
#include "stdio.h"
void timer::start(const char *message)
{
    this->start_time = std::chrono::high_resolution_clock::now();
    printf("[TIME]: start %s\n", message);
}
void timer::end(const char *message)
{
    this->end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = this->end_time - this->start_time;
    printf("[TIME]: end   %s elapse: %lf\n", message, elapsed.count());
}
timer::timer()
{
}

timer::~timer()
{
}