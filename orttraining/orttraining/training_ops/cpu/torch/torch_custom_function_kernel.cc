// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "torch_custom_function_kernel.h"
#include "core/language_interop_ops/pyop/pyop_lib_proxy.h"
#include "core/torch_custom_function/torch_custom_function_register.h"
#include <thread>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    PythonOp,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOp);

ONNX_OPERATOR_KERNEL_EX(
    PythonOpGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>()),
    PythonOpGrad);

Status PythonOp::Compute(OpKernelContext* context) const {
  ORT_ENFORCE(context);
  std::cout << "std::this_thread::get_id() in PythonOp::Compute is : " << std::this_thread::get_id() << std::endl;

  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  ORT_ENFORCE(nullptr != context);
  auto inputs_count = (size_t)ctx_internal->InputCount();
  auto outputs_count = (size_t)ctx_internal->OutputCount();
  std::vector<OrtValue*> inputs;
  std::vector<void*> outputs;

  for (size_t i = 0; i < inputs_count; ++i) {
    inputs.push_back(const_cast<OrtValue*>(ctx_internal->GetInputMLValue(i)));
  }

  auto log_func = [&](const char* msg) {
    LOGS_DEFAULT(WARNING) << msg << std::endl;
  };

  std::vector<void*> const_args;
  std::vector<int64_t> const_arg_positions;

  for (size_t i = 0; i < input_int_scalars_.size(); ++i) {
    const_arg_positions.emplace_back(input_int_scalar_positions_.at(i));
    const_args.emplace_back(Py_BuildValue("L", static_cast<long long>(input_int_scalars_.at(i))));
  }

  for (size_t i = 0; i < input_float_scalars_.size(); ++i) {
    const_arg_positions.emplace_back(input_float_scalar_positions_.at(i));
    const_args.emplace_back(Py_BuildValue("f", input_float_scalars_.at(i)));
  }

  for (size_t i = 0; i < input_int_tuple_begins_.size(); ++i) {
    // Process i-th tuple.
    // Starting index of i-th tuple in the concatenation buffer.
    const size_t begin = input_int_tuple_begins_.at(i);
    // Endding (exclusive) index of i-th tuple in the concatenation buffer.
    const size_t end = (i + 1 == input_int_tuple_begins_.size()) ? input_int_tuples_.size() : input_int_tuple_begins_.at(i + 1);
    PyObject* tuple = PyTuple_New(end - begin);
    for (size_t j = begin; j < end; ++j) {
      PyObject* item = Py_BuildValue("L", input_int_tuples_.at(j));
      PyTuple_SetItem(tuple, j - begin, item);
    }
    const_arg_positions.emplace_back(input_int_tuple_positions_.at(i));
    const_args.emplace_back(tuple);
  }

  for (size_t i = 0; i < input_float_tuple_begins_.size(); ++i) {
    // Process i-th tuple.
    // Starting index of i-th tuple in the concatenation buffer.
    const size_t begin = input_float_tuple_begins_.at(i);
    // Endding (exclusive) index of i-th tuple in the concatenation buffer.
    const size_t end = (i + 1 == input_float_tuple_begins_.size()) ? input_float_tuples_.size() : input_float_tuple_begins_.at(i + 1);
    PyObject* tuple = PyTuple_New(end - begin);
    for (size_t j = begin; j < end; ++j) {
      PyObject* item = Py_BuildValue("f", input_float_tuples_.at(j));
      PyTuple_SetItem(tuple, j - begin, item);
    }
    const_arg_positions.emplace_back(input_float_tuple_positions_.at(i));
    const_args.emplace_back(tuple);
  }

  for (size_t i = 0; i < input_pointer_scalars_.size(); ++i) {
    const_arg_positions.emplace_back(input_pointer_scalar_positions_.at(i));
    PyObject* ptr = reinterpret_cast<PyObject*>(input_pointer_scalars_.at(i));
    const_args.emplace_back(ptr);
  }

  // occupied[i] being true means the i-th input argument
  // to Python function has been set.
  std::vector<bool> occupied(inputs.size() + const_args.size(), false);

  // We know all non-tensors were set above, so let's catch up.
  for (const auto pos : const_arg_positions) {
    occupied.at(pos) = true;
  }

  // arg_positions[i] is the position index for the i-th tensor input.
  std::vector<int64_t> arg_positions;
  // Search for empty slots for tensors.
  // The i-th empty slot is assigned the i-th input tensor.
  for (size_t i = 0; i < occupied.size(); ++i) {
    if (occupied.at(i)) {
      continue;
    }
    // Find an empty slot whose position index is i.
    arg_positions.push_back(i);
  }

  std::string err;
  auto state = PyOpLibProxy::GetInstance().GetGil();
  ORT_ENFORCE(PyOpLibProxy::GetInstance().InvokePythonAutoGradFunc(instance_, "compute", inputs, arg_positions, outputs,
                                                                   log_func, const_args, const_arg_positions),
              PyOpLibProxy::GetInstance().GetLastErrorMessage(err));  //ORT_ENFORCE
  PyOpLibProxy::GetInstance().PutGil(state);

  // We had the assumption:
  // The 1st output is context index of auto grad function.
  PyObject* ctx_addr = reinterpret_cast<PyObject*>(outputs[0]);
  ORT_ENFORCE(ctx_addr, "Context object pointer should not be null");
  int64_t ctx_index = onnxruntime::python::OrtTorchFunctionPool::GetInstance().RegisterContext(ctx_addr);

  Tensor* first_output_tensor = context->Output(0, {1});
  ORT_ENFORCE(first_output_tensor != nullptr, "first_output_tensor should not be null.");
  int64_t* step_data_new = first_output_tensor->template MutableData<int64_t>();
  *step_data_new = ctx_index;

  for (size_t index = 1; index < outputs_count; ++index) {
    void* forward_ret_ortvalue_addr = outputs[index];
    // OrtValue is not release til now because we keep the values in Python side until the Python class instance is destoyed.
    // If we don't want Python do the lifecycle guarantee, we need consider PY_INCRE here as well, but be careful, need
    // operate on PyObject, directly operating on OrtValue will bring unexpected results.
    auto* forward_ret_ortvalue_ptr = reinterpret_cast<OrtValue*>(forward_ret_ortvalue_addr);
    ORT_ENFORCE(forward_ret_ortvalue_ptr != nullptr, "forward_ret_ortvalue_ptr should not be null");

    Tensor* t = forward_ret_ortvalue_ptr->GetMutable<Tensor>();
    const auto& input_shape = t->Shape();
    const auto num_dim = input_shape.NumDimensions();
    std::cout << "ortvalue addr:" << forward_ret_ortvalue_ptr << ", tenosr addr: " << t
              << ", tensor->MutableDataRaw() addr :" << reinterpret_cast<int64_t>(t->MutableDataRaw())
              << ", num_dim: " << num_dim << std::endl;

    for (size_t i = 0; i < num_dim; ++i) {
      std::cout << "PythonOp::Compute shape : " << input_shape.GetDims()[i] << std::endl;
    }

    ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(index, *forward_ret_ortvalue_ptr));
  }
  return Status::OK();
}

Status PythonOpGrad::Compute(OpKernelContext* context) const {
  ORT_ENFORCE(context);

  auto* ctx_internal = reinterpret_cast<onnxruntime::OpKernelContextInternal*>(context);
  ORT_ENFORCE(nullptr != context);
  auto inputs_count = (size_t)ctx_internal->InputCount();
  std::vector<OrtValue*> inputs;
  std::vector<void*> outputs;

  const Tensor* first_output_tensor = context->Input<Tensor>(0);
  ORT_ENFORCE(first_output_tensor != nullptr, "first_output_tensor should not be null.");
  const int64_t* context_index_ptr = first_output_tensor->template Data<int64_t>();
  PyObject* ctx_ptr = onnxruntime::python::OrtTorchFunctionPool::GetInstance().GetContext(*context_index_ptr);

  for (size_t i = 1; i < inputs_count; ++i) {
    inputs.push_back(const_cast<OrtValue*>(ctx_internal->GetInputMLValue(i)));
  }

  auto log_func = [&](const char* msg) {
    LOGS_DEFAULT(WARNING) << msg << std::endl;
  };

  std::vector<int64_t> arg_positions(inputs.size());
  for (int64_t i = 0; i < (int64_t)arg_positions.size(); ++i) {
    arg_positions.at(i) = i + 1;
  }

  std::vector<void*> const_args{ctx_ptr};
  std::vector<int64_t> const_arg_positions{0};

  std::string err;
  auto state = PyOpLibProxy::GetInstance().GetGil();
  Py_INCREF(ctx_ptr);
  ORT_ENFORCE(PyOpLibProxy::GetInstance().InvokePythonAutoGradFunc(instance_, "backward_compute", inputs, arg_positions, outputs,
                                                                   log_func, const_args, const_arg_positions),
              PyOpLibProxy::GetInstance().GetLastErrorMessage(err));
  PyOpLibProxy::GetInstance().PutGil(state);

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto* backward_ret_ortvalue_ptr = reinterpret_cast<OrtValue*>(outputs[i]);
    ORT_RETURN_IF_ERROR(ctx_internal->SetOutputMLValue(0, *backward_ret_ortvalue_ptr));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime