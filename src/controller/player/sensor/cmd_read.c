#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include "global_options.h"
#include "serial_port.h"
#include "hipnuc_dec.h"

#define RECV_BUF_SIZE 1024
#define LOG_BUF_SIZE 512

int read_start(char* port_name, int baudrate, uint8_t* recv_buf)
{
    int fd = -1;
    //uint8_t recv_buf[1024];

    if ((fd = serial_port_open(port_name)) < 0 || serial_port_configure(fd, baudrate) < 0) {
        fprintf(stderr, "Failed to open or configure port %s with %d\n", port_name, baudrate);
        return -1;
    }

    // Enable data output
    serial_send_then_recv_str(fd, "AT+EOUT=1\r\n", "OK\r\n", (char*)recv_buf, RECV_BUF_SIZE, 200);
    
    return fd;
}

void read_loop(char* port_name, int baudrate, int fd, uint8_t* recv_buf, hipnuc_raw_t *hipnuc_raw)
{
    (void)port_name;  // Unused parameter
    (void)baudrate;   // Unused parameter
    char log_buf[LOG_BUF_SIZE];

    // Read data from serial port
    int len = serial_port_read(fd, (char *)recv_buf, RECV_BUF_SIZE);
    if (len > 0) {
        
        for (int i = 0; i < len; i++) {
            // Process HipNuc data
            if (hipnuc_input(hipnuc_raw, recv_buf[i]) > 0) {
                hipnuc_dump_packet(hipnuc_raw, log_buf, LOG_BUF_SIZE);
            }
        }
    }
}

void read_end(int fd)
{
    serial_port_close(fd);
}

// int cmd_read(char* port_name, int baudrate, bool is_alive_, hipnuc_raw_t *hipnuc_raw) {
//     int fd = -1;
//     uint8_t recv_buf[1024];
//     char log_buf[512];

//     if ((fd = serial_port_open(port_name)) < 0 || serial_port_configure(fd, baudrate) < 0) {
//         log_error("Failed to open or configure port %s with %d", port_name, baudrate);
//         return -1;
//     }

//     // Enable data output
//     serial_send_then_recv_str(fd, "AT+EOUT=1\r\n", "OK\r\n", recv_buf, sizeof(recv_buf), 200);

//     // Main reading loop
//     while (is_alive_) {
//         bool new_hipnuc_data = false;

//         // Read data from serial port
//         int len = serial_port_read(fd, (char *)recv_buf, sizeof(recv_buf));
//         if (len > 0) {
            
//             for (int i = 0; i < len; i++) {
//                 // Process HipNuc data
//                 if (hipnuc_input(hipnuc_raw, recv_buf[i]) > 0) {
//                     new_hipnuc_data = true;
//                     hipnuc_dump_packet(hipnuc_raw, log_buf, sizeof(log_buf));
//                 }
//             }
//         }
//         //printf("dir: %d", hipnuc_raw->hi91.yaw);

//         // Short sleep to prevent CPU overuse
//         safe_sleep(1000);  // 1ms sleep
//     }

//     serial_port_close(fd);
//     return 0;
// }