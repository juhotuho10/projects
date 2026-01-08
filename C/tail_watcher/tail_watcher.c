// re-implementation of linux os call "tail -f file_path", where we keep watching a file 
// if it changes from os system calls 
// and if there is a change at the end, we print the last lines from file
#include <stdio.h>
#include <stdlib.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>
#include <time.h>

#define EVENT_SIZE (sizeof(struct inotify_event))
#define BUF_LEN (1024 * (EVENT_SIZE + 16))
#define MAX_LINE_LEN 4096
#define NUM_LINES 10

static char last_output[MAX_LINE_LEN * NUM_LINES];
static struct timespec last_print = {0, 0};

void print_tail(const char *filepath) {
    // prints the last NUM_LINES of file
    // if they have changed and if the update doesnt come within 10ms
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    

    long long diff_ms = (now.tv_sec - last_print.tv_sec) * 1000LL +
                        (now.tv_nsec - last_print.tv_nsec) / 1000000LL;
    if (last_print.tv_sec != 0 && diff_ms < 10) return;
    

    FILE *fp = fopen(filepath, "r");
    if (!fp) return;
    
    char *lines[NUM_LINES] = {0};
    char buffer[MAX_LINE_LEN];
    int count = 0;
    
    while (fgets(buffer, sizeof(buffer), fp)) {
        free(lines[count % NUM_LINES]);
        lines[count % NUM_LINES] = strdup(buffer);
        count++;
    }


    fclose(fp);
    
    char output[MAX_LINE_LEN * NUM_LINES] = {0};
    int total = (count < NUM_LINES) ? count : NUM_LINES;
    int start = (count < NUM_LINES) ? 0 : (count % NUM_LINES);

    
    for (int i = 0; i < total; i++) {
        int idx = (start + i) % NUM_LINES;
        if (lines[idx]) {
            strcat(output, lines[idx]);
            free(lines[idx]);
        }
    }
    

    if (strcmp(last_output, output) != 0) {
        printf("%s", output);
        fflush(stdout);
        strcpy(last_output, output);
        last_print = now;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filepath>\n", argv[0]);
        return 1;
    }
    
    char *filepath = argv[1];
    char dir_copy[256], file_copy[256];


    
    strncpy(dir_copy, filepath, sizeof(dir_copy) - 1);
    strncpy(file_copy, filepath, sizeof(file_copy) - 1);
    
    char *dir_path = dirname(dir_copy);
    char *filename = basename(file_copy);
    
    printf("watching file: %s\n", filepath);
    print_tail(filepath);
    int fd = inotify_init();
    if (fd < 0) {
        perror("inotify_init");
        return 1;
    }
    
    int wd = inotify_add_watch(fd, dir_path, IN_MODIFY);
    if (wd < 0) {
        perror("inotify_add_watch");
        close(fd);
        return 1;
    }

    
    char buffer[BUF_LEN];
    while (1) {
        // keeps checking for inotify events for the duration of the program
        int length = read(fd, buffer, BUF_LEN);
        if (length < 0) {
            perror("read");
            break;
        }
        
        for (int i = 0; i < length; ) {
            struct inotify_event *event = (struct inotify_event *)&buffer[i];
            
            if (event->len && (event->mask & IN_MODIFY) && 
                strcmp(event->name, filename) == 0) {
                print_tail(filepath);
            }
            
            i += EVENT_SIZE + event->len;
        }
    }
    
    inotify_rm_watch(fd, wd);
    close(fd);


    return 1;
}

