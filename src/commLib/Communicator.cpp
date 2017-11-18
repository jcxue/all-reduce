#include <netdb.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <new>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <unordered_map>

#include "Communicator.h"
#include "log.h"
#include "cudaLib/cuda_func.h"

#define BUF_SIZE 64
#define SOCKET_SYNC_MSG "sync"

struct PeerInfo {
	int32_t rank;
	char peerAddr[INET6_ADDRSTRLEN];
}__attribute__ ((packed));

//#######################################################################################
int32_t Communicator::get_rank()
{
    return m_rank;
}

//#######################################################################################
int32_t Communicator::is_server(const char *ip)
{
	struct ifaddrs *ifAddrStruct = nullptr;
	struct ifaddrs *ifa = nullptr;
	void * addrPtr = nullptr;
	char addressBuffer[BUF_SIZE];

	int32_t ret = 0;

	check(getifaddrs(&ifAddrStruct) == 0, "failed getifaddr");

	for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
		if (!ifa->ifa_addr) {
			continue;
		}
		if (ifa->ifa_addr->sa_family == AF_INET) { // check it is IP4
			// is a valid IP4 Address
			addrPtr = &(reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr)->sin_addr);
			inet_ntop(AF_INET, addrPtr, addressBuffer, INET_ADDRSTRLEN);
			//printf("%s IP Address %s\n", ifa->ifa_name, addressBuffer);
		} else if (ifa->ifa_addr->sa_family == AF_INET6) { // check it is IP6
			// is a valid IP6 Address
			addrPtr = &(reinterpret_cast<sockaddr_in6*>(ifa->ifa_addr)->sin6_addr);
			inet_ntop(AF_INET6, addrPtr, addressBuffer, INET6_ADDRSTRLEN);
			//printf("%s IP Address %s\n", ifa->ifa_name, addressBuffer);
		}
		if (strcmp(ip, addressBuffer) == 0) {
			ret = 1;
			break;
		}
	}

	free(ifAddrStruct);
	return ret;

error:
	if (ifAddrStruct != nullptr) {
		free(ifAddrStruct);
	}
	return -1;
}

//#######################################################################################
int32_t Communicator::create_bind_socket(const char *port) {
	struct addrinfo hints;
	struct addrinfo *result = nullptr, *rp = nullptr;
	int sockfd = -1;
	int ret = 0;

	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_family = AF_UNSPEC;
	hints.ai_flags = AI_PASSIVE;

	ret = getaddrinfo(nullptr, port, &hints, &result);
	check(ret == 0, "failed to getaddrinfo: %s", gai_strerror(ret));

	for (rp = result; rp != NULL; rp = rp->ai_next) {
		sockfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		if (sockfd < 0) {
			continue;
		}

		if (bind(sockfd, rp->ai_addr, rp->ai_addrlen) == 0) {
			/* bind success */
			break;
		}

		close(sockfd);
		sockfd = -1;
	}

	check(rp != nullptr, "no addinfo is found");
	freeaddrinfo(result);

	return sockfd;
error:
	if (result != nullptr) {
		freeaddrinfo(result);
	}
	return -1;
}

//#######################################################################################
int32_t Communicator::create_connect_socket(const char *server,
		const char *port) {
	struct addrinfo hints;
	struct addrinfo *result, *rp;
	int sock_fd = -1, ret = 0;

	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_family = AF_UNSPEC;

	ret = getaddrinfo(server, port, &hints, &result);
	check(ret == 0, "failed to getaddrinfo: %s", gai_strerror(ret));

	for (rp = result; rp != NULL; rp = rp->ai_next) {
		sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		if (sock_fd == -1) {
			continue;
		}

		ret = connect(sock_fd, rp->ai_addr, rp->ai_addrlen);
		if (ret == 0) {
			/* connection success */
			break;
		}

		close(sock_fd);
		sock_fd = -1;
	}

	check(rp!=NULL, "failed to create socket");

	freeaddrinfo(result);
	return sock_fd;

error:
	if (result) {
		freeaddrinfo(result);
	}
	if (sock_fd != -1) {
		close(sock_fd);
	}
	return -1;
}

//#######################################################################################
ssize_t Communicator::write_to_socket(int sock_fd, void *buffer, size_t len) {
	ssize_t nw, tot_written;
	const char *buf = reinterpret_cast<char *>(buffer);  // avoid pointer arithmetic on void pointer

	for (tot_written = 0; tot_written < len;) {
		nw = write(sock_fd, buf, len - tot_written);

		if (nw <= 0) {
			if (nw == -1 && errno == EINTR) {
				continue;
			} else {
				return -1;
			}
		}

		tot_written += nw;
		buf += nw;
	}
	return tot_written;
}

//#######################################################################################
ssize_t Communicator::read_from_socket(int sock_fd, void *buffer, size_t len) {
	ssize_t nr, tot_read;
	char *buf = reinterpret_cast<char *>(buffer); // avoid pointer arithmetic on void pointer
	tot_read = 0;

	while (len != 0 && (nr = read(sock_fd, buf, len)) != 0) {
		if (nr < 0) {
			if (errno == EINTR) {
				continue;
			} else {
				return -1;
			}
		}
		len -= nr;
		buf += nr;
		tot_read += nr;
	}

	return tot_read;
}

//#######################################################################################
bool Communicator::init_server(const char *server, const char *port) {
	int32_t sockfd         = 0;
	struct PeerInfo *peerInfo = nullptr;
	struct sockaddr_in peerAddr;
	socklen_t peerAddrLen = sizeof(struct sockaddr_in);
	int32_t nbytes = 0;
	char msgBuffer[BUF_SIZE] = { '\0' };

	// memory allocations
	m_sockets = new (std::nothrow) int32_t[m_numNodes];
	check(m_sockets != nullptr, "failed to allocate memory for m_sockets");

	peerInfo = new (std::nothrow) struct PeerInfo[m_numNodes];
	check (peerInfo != nullptr, "failed to allocate memory for peerInfo");

	// setup communication channel
	m_rank = 0;
	sockfd = create_bind_socket(port);
	check(sockfd > 0, "failed to create and bind socket");

	listen(sockfd, 5);

	for (int32_t i = 1; i < m_numNodes; ++i) {
		m_sockets[i] = accept(sockfd,
				reinterpret_cast<struct sockaddr *>(&peerAddr), &peerAddrLen);
		check(m_sockets[i] > 0, "failed to accept[%d]", i);

		peerInfo[i].rank = i;
		if (peerAddr.sin_family == AF_INET) {
			inet_ntop(AF_INET, &peerAddr.sin_addr, peerInfo[i].peerAddr, INET_ADDRSTRLEN);
		} else if (peerAddr.sin_family == AF_INET6) {
			inet_ntop(AF_INET6, &peerAddr.sin_addr, peerInfo[i].peerAddr, INET6_ADDRSTRLEN);
		} else {
			check(false, "peer[%d]: unknown AF type(neither ipv4 nor ipv6)", i);
		}
		log("peer[%d]: %s", i, peerInfo[i].peerAddr);
	}

	// first, tell each client its rank
	for (int32_t i = 1; i < m_numNodes; ++i) {
		int32_t rank = htonl(i);
		nbytes = write_to_socket(m_sockets[i], reinterpret_cast<char *>(&rank), sizeof(int32_t));
		check(nbytes == sizeof(int32_t), "failed to notify client[%d] its rank", i);
	}

	// wait all clients acknowledges that ranks have received
	for (int32_t i = 1; i < m_numNodes; i++) {
		nbytes = read_from_socket(m_sockets[i], msgBuffer, sizeof(SOCKET_SYNC_MSG));
		check(nbytes == sizeof(SOCKET_SYNC_MSG), "Failed to receive sync from client[%d]", i);

		if (strcmp(msgBuffer, SOCKET_SYNC_MSG) != 0) {
			check(false, "received %s instead of sync msg", msgBuffer);
		}
	}

	// second, broadcast peer info, except for server and itself, to all clients
	for (int32_t i = 1; i < m_numNodes; i++) {
		for (int32_t j = 1; j < m_numNodes; ++j) {
			if (j == i) {
				continue;
			}
			struct PeerInfo tmpInfo;
			tmpInfo.rank = htonl(peerInfo[j].rank);
			memcpy(tmpInfo.peerAddr, peerInfo[j].peerAddr, sizeof(peerInfo[j].peerAddr));

			nbytes = write_to_socket(m_sockets[i],
					reinterpret_cast<char *>(&tmpInfo),
					sizeof(struct PeerInfo));
			check(nbytes == sizeof(struct PeerInfo),
					"failed to send PeerInfo[%d] to client[%d]", j, i);
		}
		// sync with client
		nbytes = read_from_socket(m_sockets[i], msgBuffer, sizeof(SOCKET_SYNC_MSG));
		check(nbytes == sizeof(SOCKET_SYNC_MSG), "Failed to receive sync from client[%d]", i);
		if (strcmp(msgBuffer, SOCKET_SYNC_MSG) != 0) {
			check(false, "received %s instead of sync msg", msgBuffer);
		}
	}


	// clean up
	log("server initiated");
	delete []peerInfo;
	close(sockfd);
	return true;

error:

	if (peerInfo != nullptr) {
		delete []peerInfo;
	}
	if (sockfd > 0) {
		close(sockfd);
	}
	return false;
}

//#######################################################################################
bool Communicator::init_client(const char *server, const char *port) {
	int sockfd = 0;
	std::unordered_map<int32_t, std::string> peerInfo;
	struct sockaddr_in peerAddr;
	socklen_t peerAddrLen = sizeof(struct sockaddr_in);
	int32_t nbytes = 0;
	char msgBuffer[BUF_SIZE] = { '\0' };

	sockfd = create_bind_socket(port);
	check(sockfd > 0, "failed to create and bind socket");
	listen(sockfd, 5);

	// memory allocations
	m_sockets = new (std::nothrow) int32_t[m_numNodes];
	check(m_sockets != nullptr, "failed to allocate memory for m_sockets");

	// connect to server
	m_sockets[0] = create_connect_socket(server, port);
	check(m_sockets[0] > 0, "failed to connect to server");

	// get rank from server
	nbytes = read_from_socket(m_sockets[0], reinterpret_cast<char *>(&m_rank), sizeof(m_rank));
	check(nbytes == sizeof(m_rank), "Failed to receive rank from server");
	m_rank = ntohl(m_rank);
	check(m_rank > 0, "invalid rank value: %d", m_rank);
	log("rank = %d", m_rank);

	// notify server the rank info has received
	sprintf(msgBuffer, "%s", SOCKET_SYNC_MSG);
	nbytes = write_to_socket(m_sockets[0], msgBuffer, sizeof(SOCKET_SYNC_MSG));
	check(nbytes == sizeof(SOCKET_SYNC_MSG), "Failed to sync with server");

	// each client setups new connections to peers that have rank
	// smaller than its own rank except to master which has already
	// been established
	for (int32_t i = 0; i < (m_numNodes - 2); ++i) {
		struct PeerInfo tmpInfo;
		nbytes = read_from_socket(m_sockets[0],
				reinterpret_cast<char *>(&tmpInfo), sizeof(tmpInfo));
		check(nbytes == sizeof(tmpInfo), "failed to read peer info from server");

		int32_t rank = ntohl(tmpInfo.rank);
		check(rank > 0, "invalid rank value %d for %s", rank, tmpInfo.peerAddr);
		log("receive peer[%d]: %s", rank, tmpInfo.peerAddr);

		std::unordered_map<int32_t, std::string>::iterator it;
		it = peerInfo.find(rank);
		if (it != peerInfo.end()) {
			check(false, "%s has the same rank as %s", tmpInfo.peerAddr, it->second.c_str());
		} else {
			peerInfo[rank] = std::string(tmpInfo.peerAddr);
		}
	}

	// sync with server
	nbytes = write_to_socket(m_sockets[0], msgBuffer, sizeof(SOCKET_SYNC_MSG));
	check(nbytes == sizeof(SOCKET_SYNC_MSG), "Failed to sync with server");

	// connect to the peers with lower ranks
	for (int32_t i = 1; i < m_rank; ++i) {
		log("try to connect %d (%s)", i, peerInfo[i].c_str());
		m_sockets[i] = create_connect_socket(peerInfo[i].c_str(), port);
		check(m_sockets[i] > 0, "failed to connect to %s", peerInfo[i].c_str());
		log("connected to %d (%s)", i, peerInfo[i].c_str());
	}

	// accept connections from peers with higher ranks
	for (int32_t i = (m_rank+1); i < m_numNodes; ++i) {
		char addrStr[INET6_ADDRSTRLEN];
		int32_t tmpSockfd = accept(sockfd, reinterpret_cast<struct sockaddr *>(&peerAddr), &peerAddrLen);
		check(tmpSockfd > 0, "failed to accept[%d]", i);

		if (peerAddr.sin_family == AF_INET) {
			inet_ntop(AF_INET, &peerAddr.sin_addr, addrStr, INET_ADDRSTRLEN);
		} else if (peerAddr.sin_family == AF_INET6) {
			inet_ntop(AF_INET6, &peerAddr.sin_addr, addrStr, INET6_ADDRSTRLEN);
		} else {
			check(false, "peer[%d]: unknown AF type(neither ipv4 nor ipv6)", i);
		}

		// find the rank
		std::unordered_map<int32_t, std::string>::iterator it;
		int32_t rank = -1;
		for(it = peerInfo.begin(); it != peerInfo.end(); ++it) {
			if(strcmp(it->second.c_str(), addrStr) == 0) {
				rank = it->first;
				break;
			}
		}
		check(rank > m_rank, "%s has rank %d less than %d", addrStr, rank, m_rank);
		m_sockets[rank] = tmpSockfd;
		log("received connection from %d (%s)", rank, addrStr);
	}

	log("client initiated");
	close(sockfd);

	return true;

error:

	if (sockfd > 0) {
		close(sockfd);
	}
	return false;
}

//#######################################################################################
bool Communicator::init(const char *serverIP, const char *numNodes, const char *port) {
	int32_t ret = is_server(serverIP);
	check(ret >= 0, "failed to perform server check");

	m_numNodes = atoi(numNodes);
	
	if (ret == 1) {
		return init_server(serverIP, port);
	} else {
		return init_client(serverIP, port);
	}

error:
	return false;
}

//#######################################################################################
void Communicator::all_reduce(float *x, float *y, int64_t n) {
    ssize_t ret = 0;
    size_t len = n * sizeof(float);

    if (m_rank != m_numNodes-1) {
	ret = read_from_socket(m_sockets[m_rank+1], y, len);
	if (ret != len) {
	    log("Received %zd (expecting %zu)", ret, len);
	}
    }

    gpu_reduce(n, x, y);

    if (m_rank > 0) {
	ret = write_to_socket(m_sockets[m_rank-1], y, len);
	if (ret != len) {
	    log("Received %zd (expecting %zu)", ret, len);
	}
    }
}

//#######################################################################################
Communicator::~Communicator() {
	if (m_sockets != nullptr) {
		delete []m_sockets;
	}
}
