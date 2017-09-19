#include "buffer_bank.h"

namespace horovod {
namespace tensorflow {

SharpBuf::SharpBuf(size_t buffer_size_, struct sharp_coll_context ctx_): ctx(ctx_){
  buf = (void*) malloc(buffer_size_ * sizeof(char));
  int res = sharp_coll_reg_mr(ctx_, buf, buffer_size_, &mr);
  if (res < 0){
     free(buf);
     buf = NULL;
  }
}

SharpBuf::~SharpBuf(){
  int res = sharp_coll_dereg_mr(ctx, mr);
  if (res < 0){
    throw res;
  }
  free(buf);
}

BufferBank::BufferBank(size_t buffer_size_,  struct sharp_coll_context ctx_): buffer_size(buffer_size_), count(0), ctx(ctx_), buffers(), freelist(), map(){

}

SharpBuf* BufferBank::request(uint16_t idx){
  if (freelist.empty()){
    this->expand();
  }
  size_t next_free = freelist.front();
  freelist.pop();
  map[idx] = next_free;
  return buffers[next_free];
}

SharpBuf* BufferBank::expand(){
  SharpBuf* new_buf = new SharpBuf(buffer_size , ctx);
  buffers.insert(buffers.end(), new_buf);
  freelist.push(count);
  ++count;
}

void BufferBank::release(uint16_t idx){
  std::map<uint16_t, size_t>::iterator it = map.find(idx);
  size_t buf_num = it->second;
  freelist.push(buf_num);
  map.erase(it);
}

BufferBank::~BufferBank(){
  for (std::vector<SharpBuf*>::iterator it = buffers.begin(); it != buffers.end(); ++it){
    free(*it);
  }
}


}}
