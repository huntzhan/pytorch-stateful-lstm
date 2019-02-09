#include "extension/unidirectional_lstm.h"
#include "extension/stateful_unidirectional_lstm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch::python::bind_module<cnt::UnidirectionalSingleLayerLstmImpl>(
      m, "UnidirectionalSingleLayerLstm")

      .def(
          py::init<
              int64_t, int64_t, int64_t,
              bool, int64_t,
              double, double,
              int64_t, double>(),
          // Required.
          py::arg("input_size"),
          py::arg("hidden_size"),
          py::arg("cell_size"),
          // Optional.
          py::arg("go_forward") = true,
          py::arg("truncated_bptt") = 0,
          py::arg("cell_clip") = 0.0,
          py::arg("proj_clip") = 0.0,
          py::arg("recurrent_dropout_type") = 0,
          py::arg("recurrent_dropout_probability") = 0.0)

      .def(
          "cuda",
          [](cnt::UnidirectionalSingleLayerLstmImpl& module, int64_t device) {
            module.to("cuda:" + std::to_string(device));
          })

      .def(
          "__call__",
          &cnt::UnidirectionalSingleLayerLstmImpl::forward);

  torch::python::bind_module<cnt::UnidirectionalLstmImpl>(
      m, "UnidirectionalLstm")

      .def(
          py::init<
              int64_t,
              int64_t, int64_t, int64_t,
              bool, int64_t,
              double, double,
              int64_t, double,
              bool>(),
          // Required.
          py::arg("num_layers"),
          py::arg("input_size"),
          py::arg("hidden_size"),
          py::arg("cell_size"),
          // Optional.
          py::arg("go_forward") = true,
          py::arg("truncated_bptt") = 0,
          py::arg("cell_clip") = 0.0,
          py::arg("proj_clip") = 0.0,
          py::arg("recurrent_dropout_type") = 0,
          py::arg("recurrent_dropout_probability") = 0.0,
          py::arg("use_skip_connections") = false)

      .def(
          "cuda",
          [](cnt::UnidirectionalLstmImpl& module, int64_t device) {
            module.to("cuda:" + std::to_string(device));
          })

      .def(
          "__call__",
          &cnt::UnidirectionalLstmImpl::forward);

  torch::python::bind_module<cnt::StatefulUnidirectionalLstmImpl>(
      m, "StatefulUnidirectionalLstm")

      .def(
          py::init<
              int64_t,
              int64_t, int64_t, int64_t,
              bool, int64_t,
              double, double,
              int64_t, double,
              bool>(),
          // Required.
          py::arg("num_layers"),
          py::arg("input_size"),
          py::arg("hidden_size"),
          py::arg("cell_size"),
          // Optional.
          py::arg("go_forward") = true,
          py::arg("truncated_bptt") = 0,
          py::arg("cell_clip") = 0.0,
          py::arg("proj_clip") = 0.0,
          py::arg("recurrent_dropout_type") = 0,
          py::arg("recurrent_dropout_probability") = 0.0,
          py::arg("use_skip_connections") = false)

      .def(
          "cuda",
          [](cnt::StatefulUnidirectionalLstmImpl& module, int64_t device) {
            module.to("cuda:" + std::to_string(device));
          })

      .def(
          "__call__",
          &cnt::StatefulUnidirectionalLstmImpl::forward)

      .def(
          "permutate_states",
          &cnt::StatefulUnidirectionalLstmImpl::permutate_states)

      .def(
          "reset_states",
          &cnt::StatefulUnidirectionalLstmImpl::reset_states)

      .def(
          "managed_hidden_state",
          &cnt::StatefulUnidirectionalLstmImpl::managed_hidden_state)

      .def(
          "managed_cell_state",
          &cnt::StatefulUnidirectionalLstmImpl::managed_cell_state);
}
