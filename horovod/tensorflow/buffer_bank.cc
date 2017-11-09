#include "buffer_bank.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#define PUT_ON_GPU
#define DATA_SIZE 4


using perftools::gputools::Stream;



namespace horovod {
namespace tensorflow {

SharpSpec::SharpSpec(size_t buffer_size_, enum sharp_datatype dtype_ ,struct sharp_coll_context* ctx_, OpKernelContext* op_ctx): ctx(ctx_){
  int byte_size = buffer_size_;
  int num_elements = byte_size / DATA_SIZE;
  specs.sbuf_desc.type = SHARP_DATA_BUFFER;
  specs.rbuf_desc.type = SHARP_DATA_BUFFER;
//  printf("Allocating new sharp spec...\n");

#ifndef PUT_ON_GPU
  void* buf = (void*) malloc(byte_size * sizeof(char));
#else

  TensorShape buffer_shape;
  buffer_shape.AddDim(num_elements);
  buffer = new PersistentTensor();

  Tensor* tensor;

  if (!op_ctx){
    printf("no op context!!!\n");
  } 

  Status status;

  AllocatorAttributes attr = AllocatorAttributes();
  attr.set_gpu_compatible(true);
  attr.set_on_host(false);
  attr.set_nic_compatible(false);

  status = op_ctx->allocate_persistent(DT_FLOAT, buffer_shape, buffer, &tensor, attr);


  if (!status.ok()) {
    printf("Persistent Tensor Allocation Failed!\n");
    return;
  }

#if HAVE_CUDA
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = op_ctx->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
  void* buf = DMAHelper::base(buffer->AccessTensor(op_ctx));
#endif
#endif

  specs.sbuf_desc.buffer.ptr = buf;
  int res = sharp_coll_reg_mr(ctx_, buf, byte_size , &specs.sbuf_desc.buffer.mem_handle);

//  printf("GPU Buffer registered with SHARP, affr = %d, size = %d\n", buf, buffer_size_);

  specs.sbuf_desc.buffer.length = buffer_size_;
  if (res < 0){
     free(buf);
     buf = NULL;
     return;
  }
  //specs.rbuf_desc = specs.sbuf_desc; //in place
  
  specs.rbuf_desc.buffer.ptr = (void*) malloc(byte_size);
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
  free(buffer);
  free(specs.rbuf_desc.buffer.ptr);
#ifndef PUT_ON_GPU
  free(specs.sbuf_desc.buffer.ptr);
#endif
}

struct sharp_coll_reduce_spec* SharpSpec::spec(){
  return &specs;
}

struct sharp_coll_reduce_spec* SharpSpec::spec(int len){
  specs.length = len/DATA_SIZE;
  return &specs;
}

void SharpSpec::set_length(int len){
  specs.length = len;
}

void* SharpSpec::sbuf() const{
  return specs.sbuf_desc.buffer.ptr;
}

void* SharpSpec::rbuf() const{
  return specs.rbuf_desc.buffer.ptr;
}

cudaStream_t* SharpSpec::stream(){
  return &mstream;
}

BufferBank::BufferBank(): buffer_size(0), count(0), buffers(), freelist(), map(), initiated(false){}

void BufferBank::Init(size_t buffer_size_,  struct sharp_coll_context* ctx_, enum sharp_datatype dtype_){
  buffer_size = buffer_size_;
  ctx = ctx_;
  dtype = dtype_;
  initiated = true;
}

SharpSpec* BufferBank::request(uint16_t idx, OpKernelContext* op_ctx){
  if (freelist.empty()){
    this->expand(op_ctx);
  }
  size_t next_free = freelist.front();
  freelist.pop();
//  printf("buffer %d being used\n", next_free);
  map[idx] = next_free;
  return buffers[next_free];
}

void BufferBank::expand(OpKernelContext* op_ctx){
  SharpSpec* new_spec = new SharpSpec(buffer_size , dtype , ctx, op_ctx);
  buffers.insert(buffers.end(), new_spec);
  freelist.push(count);
  ++count;
}

void BufferBank::release(uint16_t idx){
  std::map<uint16_t, size_t>::iterator it = map.find(idx);
  size_t buf_num = it->second;
  freelist.push(buf_num);
//  printf("buffer %d released\n", buf_num);
  map.erase(it);
}

BufferBank::~BufferBank(){
  for (std::vector<SharpSpec*>::iterator it = buffers.begin(); it != buffers.end(); ++it){
    free(*it);
  }
}

bool BufferBank::isInitiated() const{
  return initiated;
}

}}
