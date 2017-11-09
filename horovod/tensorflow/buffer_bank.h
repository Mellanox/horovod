
#ifdef HAVE_SHARP

#ifndef TF_BUFFERBANK_H
#define TF_BUFFERBANK_H

#include "mpi_message.h"
#include "timeline.h"
#include <map>
#include <thread>
#include <queue>
#include <string.h>
#include <assert.h>
#include <sharp/api/sharp.h>
#include "tensorflow/stream_executor/stream.h"

#include "tensorflow/core/framework/op_kernel.h"

#include <cuda_runtime.h>

namespace horovod {
namespace tensorflow {


class SharpSpec{
public:
  SharpSpec(size_t buffer_size_, enum sharp_datatype dtype_  , struct sharp_coll_context* ctx_, OpKernelContext* op_ctx);
  ~SharpSpec();

  struct sharp_coll_reduce_spec* spec();
  struct sharp_coll_reduce_spec* spec(int len);
  void set_length(int len);

  void* rbuf() const;
  void* sbuf() const;

  cudaStream_t* stream();

private:
  struct sharp_coll_context *ctx;
  struct sharp_coll_reduce_spec specs;
  PersistentTensor* buffer;

  cudaStream_t mstream;
};

class BufferBank{
 public:
  BufferBank();
  ~BufferBank();
  SharpSpec* request(uint16_t idx, OpKernelContext* op_ctx);
  void release(uint16_t idx);
  
  void Init(size_t buffer_size_,  struct sharp_coll_context* ctx_ , enum sharp_datatype dtype_ = SHARP_DTYPE_FLOAT);
  bool isInitiated() const;

 private:
  void expand(OpKernelContext* op_ctx);
  size_t buffer_size;  
  size_t count;
  struct sharp_coll_context *ctx;
  std::vector<SharpSpec*> buffers;
  std::queue<size_t> freelist;
  std::map<uint16_t, size_t> map;

  enum sharp_datatype dtype;

  bool initiated;
};



}
}

#endif
#endif
