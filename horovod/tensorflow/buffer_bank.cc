#include "buffer_bank.h"

namespace horovod {
namespace tensorflow {

SharpSpec::SharpSpec(size_t buffer_size_, enum sharp_datatype dtype_ ,struct sharp_coll_context ctx_): ctx(ctx_){
  specs.sbuf_desc.type = SHARP_DATA_BUFFER;
  void* buf = (void*) malloc(buffer_size_ * sizeof(char));  //TODO: Put on GPU
  specs.sbuf_desc.buffer.ptr = buf;
  int res = sharp_coll_reg_mr(ctx_, buf, buffer_size_, &specs.sbuf_desc.buffer.mem_handle);
  specs.sbuf_desc.buffer.length = buffer_size_;
  if (res < 0){
     free(buf);
     buf = NULL;
     return;
  }
  specs.rbuf_desc = specs.sbuf_desc; //in place
  specs.root = 0; //ignored
  specs.dtype = dtype_;
  specs.length = -1;
  specs.op = SHARP_OP_SUM;
}

SharpSpec::~SharpSpec(){
  int res = sharp_coll_dereg_mr(ctx, specs.sbuf_desc.buffer.mem_handle);
  if (res < 0){
    throw res;
  }
  free(specs.sbuf_desc.buffer.ptr);
}

struct sharp_coll_reduce_spec* SharpSpec::spec() const{
  return &specs;
}

struct sharp_coll_reduce_spec* SharpSpec::spec(int len) const{
  specs.length = len;
  return &specs;
}

void* SharpSpec::sbuf() const{
  return specs.sbuf_desc.buffer.ptr;
}

void* SharpSpec::rbuf() const{
  return specs.rbuf_desc.buffer.ptr;
}

void SharpSpec::set_length(int len) const{
  specs.length = len;
}


BufferBank::BufferBank(size_t buffer_size_,  struct sharp_coll_context ctx_, enum sharp_datatype dtype_ = SHARP_DTYPE_FLOAT): 
                       buffer_size(buffer_size_), count(0), ctx(ctx_), buffers(), freelist(), map(), dtype(dtype_){


}

SharpSpec* BufferBank::request(uint16_t idx){
  if (freelist.empty()){
    this->expand();
  }
  size_t next_free = freelist.front();
  freelist.pop();
  map[idx] = next_free;
  return buffers[next_free];
}

SharpSpec* BufferBank::expand(){
  SharpSpec* new_spec = new SharpSpec(buffer_size , ctx);
  buffers.insert(buffers.end(), new_spec);
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
  for (std::vector<SharpSpec*>::iterator it = buffers.begin(); it != buffers.end(); ++it){
    free(*it);
  }
}


}}
