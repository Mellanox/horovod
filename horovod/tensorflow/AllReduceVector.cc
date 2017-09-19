#include "AllReduceVector.h"

//#define NUM_RESERVED 1

enum MReduce {
  MR_EXTEND,
  MR_FINISH,
  NUM_RESERVED
};

#define MR_bit(y) (size_*8-(1+y))

namespace horovod {
namespace tensorflow {

AllReduceVector::AllReduceVector(int bits): cmpl(0), lastRound(false), pending(0), timeline(NULL){
  size_ = (bits + NUM_RESERVED )/8 + (((bits + NUM_RESERVED)%8)?1:0);
  vec = new uint8_t[size_]; 
  memset(vec,0,size_);
}

AllReduceVector::~AllReduceVector(){
  for (MsgIt it = msgs.begin(); it != msgs.end(); ++it){
     free(it->second);
  }

  while (!ready_q.empty()){
    free(ready_q.front());    ready_q.pop();
  }

  delete[](vec);
}

bool AllReduceVector::add(IdReq idreq){
  int32_t& idx = idreq.first;
  if (vec[idx/8] & (uint8_t) (1 << (idx%8))){
    return false;
  } else {
    ++pending;
    msgs.insert(idreq);
    vec[idx/8]|= (1 << (idx%8));
    return true;
  }
}

int AllReduceVector::check(uint8_t* buf){
  MsgIt it;
  uint8_t c;
  int k;
  int cnt = 0;
  for (int i = (size_ -1) ; i >= 0; --i){
    c = buf[i];
    k = 7;
    while (c){
      if (c & 128){
        int32_t mbit = k+8*i;
        it = msgs.find(mbit);
        if (it == msgs.end()){
          switch (8*size_-1-mbit){
            case MR_FINISH:
              lastRound = true;
              break;
            default: 
              printf("Error: msg %d not found. Messages at:\n", mbit);
              for (MsgIt it = msgs.begin(); it != msgs.end(); ++it){
                printf("%d\t",it->first);
              }
              printf("\n");
          }
        } else {
          ++cnt;
          if (timeline){
            timeline->NegotiateEnd(it->second->tensor_name());
          }
          ready_q.push(it->second);
          msgs.erase(it);
        }
      }
      --k;
      c = c << 1;
    }
    vec[i]^= buf[i];
  }
  pending-=cnt;
  return cnt;
}



MPIRequest* AllReduceVector::pop(){
  MPIRequest* res = ready_q.front();
  ready_q.pop();
  return res;
}

int AllReduceVector::num_ready() const {
  return ready_q.size();
}

void AllReduceVector::addAll(){
  while (!pre_q.empty()){
    IdReq idreq = pre_q.front();
    if (timeline){
      MPIRequest* msg = idreq.second;
      timeline->NegotiateStart(msg->tensor_name(), msg->request_type());
    }
    pre_q.pop();
    add(idreq);
  }
}

void AllReduceVector::insert(int32_t idx, MPIRequest* req){
  pre_q.push(IdReq(idx,req));
}

void AllReduceVector::markLast(){
  uint32_t idx = MR_bit(MR_FINISH);
  vec[idx/8]|= (1 << (idx%8));
  ++pending;
}

}}
