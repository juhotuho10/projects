// Program for getting all the files in a folder
// and then listing different data for the files


#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <string.h>
#include <time.h>
#include <grp.h>
#include <dirent.h> 
#include <fcntl.h> 

const char* string_time(time_t t) {
    // unix time -> string time
    static char buf[64];
    struct tm* tm = localtime(&t);

    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    
    return buf;
}

char* file_type(mode_t mode) {
    // checks the file type and returns the file type
    if (S_ISREG(mode)) return "file";
    if (S_ISLNK(mode)) return "link";
    if (S_ISDIR(mode)) return "directory";
    if (S_ISCHR(mode)) return "character device";
    if (S_ISBLK(mode)) return "block device";
    if (S_ISFIFO(mode)) return "pipe";
    if (S_ISSOCK(mode)) return "socket";
    return "?";
}



int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s directory_path\n", argv[0]);
        return 1;
    }

    DIR* directory;
    struct dirent* dir;
    directory = opendir(argv[1]);
    if (directory) {
        printf("file information:\n");
        while ((dir = readdir(directory)) != NULL) {
            // skip the current directory and the parent directory

            if ((strcmp(dir->d_name, ".") == 0) || (strcmp(dir->d_name, "..") == 0)) continue;


            // concatenate the path and the filename
            char fullpath[128];
            snprintf(fullpath, sizeof(fullpath), "%s/%s", argv[1], dir->d_name);

            struct stat statistics;
            // reads the file information
            int err = lstat(fullpath, &statistics);
            if (err == -1) {
                // cannot read statistics from the file or it does not exist
                perror("statistics");

                continue;
            }
            // file information
            printf("name : %s\n", dir->d_name);
            printf("\tbyte size : %d\n", (long)statistics.st_size);
            printf("\tuid : %d\n", (long)statistics.st_uid);
            printf("\tmodification time : %s\n", string_time(statistics.st_mtime));
            printf("\tpermissions: (%o)\n", statistics.st_mode & 07777);
            printf("\ttype: %s\n", file_type(statistics.st_mode));
            


            // user's file comment from xattribute
            char comment_buf[256];
            ssize_t size = getxattr(fullpath, "user.comment", comment_buf, sizeof(comment_buf) - 1);

            if (size >= 0) {
                comment_buf[size] = '\0';
                printf("\tuser comment: %s\n", comment_buf);
            }
            else {
                printf("\tno user comment\n");
            }


            printf("\n");
        }
        closedir(directory);

    }
    return(0);
}
