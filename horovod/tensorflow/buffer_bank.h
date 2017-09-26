
#ifdef HAVE_SHARP

#ifndef TF_BUFFERBANK_H
#define TF_BUFFERBANK_H

#include "mpi_message.h"
#include "timeline.h"
#include <map>
#include <queue>
#include <string.h>
#include <assert.h>
#include <sharp/api/sharp.h>

namespace horovod {
namespace tensorflow {


struct SharpBuf{
  SharpBuf(size_t buffer_size_, struct sharp_coll_context ctx_);
  ~SharpBuf();

  struct sharp_coll_context *ctx;

  void* buf;
  size_t length;
  void* mr;
}

class BufferBank{
 public:
  BufferBank(size_t buffer_size_,  struct sharp_coll_context ctx_ );
  ~BufferBank();
  SharpBuf* request(uint16_t idx);
  void release(uint16_t idx);
 private:
  void expand();
  size_t buffer_size;  
  size_t count;
  struct sharp_coll_context *ctx;
  std::vector<SharpBuf*> buffers;
  std::queue<size_t> freelist;
  std::map<uint16_t, size_t> map;
};


}
}

#endif
#endif
