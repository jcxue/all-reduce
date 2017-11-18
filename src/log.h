#ifndef LOG_H_
#define LOG_H_

#include <errno.h>
#include <stdio.h>

#define log(M, ...) fprintf(stderr, "" M "\n", ##__VA_ARGS__)
#define logError(M, ...) fprintf(stderr, "\033[1;31m[ERROR]\033[0m " M "\n", ##__VA_ARGS__)
#define check(A, M, ...) if(!(A)) {logError(M, ##__VA_ARGS__); errno=0; goto error;}

#endif /* log.h */
