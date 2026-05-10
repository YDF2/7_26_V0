// command_handlers.h
#ifndef COMMAND_HANDLERS_H
#define COMMAND_HANDLERS_H

#include "global_options.h"
#include <stdbool.h>
#include <string.h>
#include "hipnuc_dec.h"

#ifdef __cplusplus
extern "C" {
#endif

//int cmd_read(char* port_name, int baudrate, bool is_alive_, hipnuc_raw_t *hipnuc_raw);
int read_start(char* port_name, int baudrate, uint8_t* recv_buf);
void read_loop(char* port_name, int baudrate, int fd, uint8_t* recv_buf, hipnuc_raw_t *hipnuc_raw);
void read_end(int fd);

#ifdef __cplusplus
}
#endif

#endif // COMMAND_HANDLERS_H