// Header for utils.cpp, a little library of low-level array and timer stuff.
// (rest of finufft defs and types are now in defs.h)

#ifndef UTILS_H
#define UTILS_H

// jfm's timer class
#include <sys/time.h>
class CNTime {
 public:
  void start();
  double restart();
  double elapsedsec();
 private:
  struct timeval initial;
};

#endif  // UTILS_H
