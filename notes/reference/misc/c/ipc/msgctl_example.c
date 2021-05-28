#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

int main(void)
{
	int msqid;
	key_t key;
	struct msqid_ds msqid_ds, *buf;
	buf = & msqid_ds;

	if ((key = ftok("msgsnd_example.c", 'B')) == -1) {
		perror("ftok");
		exit(1);
	}

	if ((msqid = msgget(key, 0644)) == -1) {
		perror("msgget");
		exit(1);
	}

	msgctl(msqid, IPC_STAT, buf);
	printf("\nThe user id = %d\n", buf->msg_perm.uid);
	printf("The group id = %d\n", buf->msg_perm.gid);
	printf("The operation permissions = 0%o\n", buf->msg_perm.mode);
	printf("The msg_qbytes = %d\n", buf->msg_qbytes);

	buf->msg_perm.gid = 20;
	msgctl(msqid, IPC_SET, buf);

	msgctl(msqid, IPC_STAT, buf);
	printf("\nThe user id = %d\n", buf->msg_perm.uid);
	printf("The group id = %d\n", buf->msg_perm.gid);
	printf("The operation permissions = 0%o\n", buf->msg_perm.mode);
	printf("The msg_qbytes = %d\n", buf->msg_qbytes);

	exit(0);
}
