
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


class SharpSpec{
public:
  SharpDesc(size_t buffer_size_, enum sharp_datatype dtype_  , struct sharp_coll_context ctx_);
  ~SharpDesc();


  struct sharp_coll_reduce_spec* spec() const;
  struct sharp_coll_reduce_spec* spec(int len);
  void set_length(int len);

  void* rbuf() const;
  void* sbuf() const;

private:
  struct sharp_coll_context *ctx;
  struct sharp_coll_reduce_spec specs;
}

class BufferBank{
 public:
  BufferBank(size_t buffer_size_,  struct sharp_coll_context ctx_ , enum sharp_datatype dtype_ );
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
  enum sharp_datatype dtype;
};


}
}

#endif
#endif
