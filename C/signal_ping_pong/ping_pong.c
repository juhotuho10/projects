// Program that forks itself N times, and plays ping-pong with the forked processes
// until the forks are killed and the process exits

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>

#define MAX_KOPIOITA 128

int got_signals = 0;
pid_t children_pids[MAX_KOPIOITA];
int copies = 0;

void handler_1(int _sig) {
    printf("ping\n");
    got_signals++;
}

void handler_2(int _sig) {
    printf("pong\n");
}

int main(int argc, char *argv[]) {
    if (argc != 3 || strcmp(argv[1], "--copies") != 0) {
        fprintf(stderr, "usage: %s --copies N\n", argv[0]);
        return 1;
    }

    copies = atoi(argv[2]);
    if (copies <= 0 || copies > MAX_KOPIOITA) {
        fprintf(stderr, "N copies must be between 1 - %d\n", MAX_KOPIOITA);
        return 1;
    }

    struct sigaction sig_action_1 = {0};
    struct sigaction sig_action_2 = {0};
    sig_action_1.sa_handler = handler_1;
    sig_action_2.sa_handler = handler_2;

    sigaction(SIGUSR1, &sig_action_1, NULL);
    sigaction(SIGUSR2, &sig_action_2, NULL);

    pid_t parent_pid = getpid();

    for (int i = 0; i < copies; i++) {
        pid_t pid = fork();
        if (pid == 0) { // pid 0 i.e. child process
            kill(parent_pid, SIGUSR1); // signal SIGUSR1 to the parent
            pause(); // wait for a response
            _exit(0); 
        } else if (pid > 0) { // parent process
            children_pids[i] = pid;
            usleep(100000); // wait for the child process to be ready
        } else {
            perror("fork");
            return 1;
        }
    }

    

    // wait for all child processes before continuing
    while (got_signals < copies)
        pause();

    // signal SIGUSR2 to all child processes 
    for (int i = 0; i < copies; i++)
        kill(children_pids[i], SIGUSR2);


    // wait for all child processes to terminate
    for (int i = 0; i < copies; i++)
        waitpid(children_pids[i], NULL, 0);

    printf("done\n");
    return 0;
}

