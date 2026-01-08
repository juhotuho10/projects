# File tail watcher

re-implementation of linux's "tail -f file_path", where we keep watching a file and if it changes from os system calls 
and if there is a change at the end, we print the last lines from file

Done as a part of Operating Systems course to learn more about sytemcalls and their usage

Running the program requires argument: `path_to_file`

Can be built just by running `make`

Requires Linux for the systemcalls
