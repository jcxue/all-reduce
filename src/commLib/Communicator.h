#ifndef COMMUNICATOR_H_
#define COMMUNICATOR_H_

#include <inttypes.h>
#include <string>

class Communicator {
public:

	~Communicator();

	bool init(const char *serverIP, const char *numNodes, const char *port);
	int32_t get_rank();
	void all_reduce(float *x, float *y, int64_t n);
	
private:

	int32_t m_rank;
	int32_t m_numNodes;
	int32_t *m_sockets; // sockets used to communicate with peers

	int32_t is_server(const char *ip);

	int32_t create_bind_socket(const char *port);
	int32_t create_connect_socket(const char *server, const char *port);

	ssize_t read_from_socket(int sock_fd, void *buffer, size_t len);
	ssize_t write_to_socket(int32_t sock_fd, void *buffer, size_t len);

	bool init_server(const char *serverIP, const char *port);
	bool init_client(const char *serverIP, const char *port);

};

#endif /* communicator.h */
