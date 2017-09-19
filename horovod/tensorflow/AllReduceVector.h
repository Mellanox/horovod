


#ifndef TF_ALLREDUCEVECTOR_H
#define TF_ALLREDUCEVECTOR_H

#include "mpi_message.h"
#include "timeline.h"
#include <map>
#include <queue>
#include <string.h>
#include <assert.h>

namespace horovod {
namespace tensorflow {

typedef std::map<int32_t, MPIRequest*> MsgTable;
typedef std::map<int32_t, MPIRequest*>::iterator MsgIt;

typedef std::pair<int32_t,MPIRequest*> IdReq;

class AllReduceVector{

 public:
  
  AllReduceVector(int bits);
  AllReduceVector() : AllReduceVector(500) {}

  ~AllReduceVector();

  bool add(IdReq idreq );
  
  int check(uint8_t* buf);

  int size() const {return size_;}

  uint8_t* buf() const {return vec;}

  void addAll();

  void insert(int32_t idx,  MPIRequest* req);

  bool isLast() const {return lastRound;}

  void markLast();

  bool hasWork() {return pending;}

  MPIRequest* pop();
  int num_ready() const;

  Timeline* timeline;

 private:
  int size_;  //bytes size_ 
  uint8_t* vec;
  MsgTable msgs;
  int cmpl;  
  std::queue<MPIRequest*> ready_q;
  std::queue<IdReq> pre_q;  //shared by multipile processes. protected by the state mutex
  bool lastRound;
  int pending;
};


}
}

#endif
